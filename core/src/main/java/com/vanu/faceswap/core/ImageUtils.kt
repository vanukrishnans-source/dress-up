package com.vanu.faceswap.core

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ColorSpace
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.max
import kotlin.math.roundToInt

/** Image loading failed; [message] is user-facing, [cause] carries the technical reason. */
class ImageLoadException(message: String, cause: Throwable? = null) : IOException(message, cause)

object ImageUtils {
    const val MAX_SIDE = 2048
    const val TAG = "FaceSwap"

    /**
     * Load a picked/captured image: copy it to app cache first (Photo Picker grants are temporary and
     * some providers give one-shot streams), then decode with downscaling and EXIF orientation applied.
     */
    fun load(context: Context, uri: Uri, slot: String, maxSide: Int = MAX_SIDE): Bitmap {
        val file = copyToCache(context, uri, slot)
        try {
            return loadFile(file, maxSide)
        } catch (e: ImageLoadException) {
            // Last resort on Android 10+: let the media provider / photo picker decode it for us
            // (it can render HEIC/AVIF and camera formats even if our decoders refused the file).
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                try {
                    val t = context.contentResolver.loadThumbnail(uri, android.util.Size(maxSide, maxSide), null)
                    Log.w(TAG, "decoders failed; using provider-rendered image ${t.width}x${t.height}", e)
                    return fitWithin(ensureArgb8888(t), maxSide)
                } catch (t: Throwable) {
                    e.addSuppressed(t)
                }
            }
            throw e
        }
    }

    /** Copy the content behind [uri] into cacheDir/picked. Each read uses a fresh stream. */
    fun copyToCache(context: Context, uri: Uri, slot: String): File {
        val dir = File(context.cacheDir, "picked").apply { mkdirs() }
        dir.listFiles()?.filter { it.name.startsWith("${slot}_") }?.forEach { it.delete() }
        val out = File(dir, "${slot}_${System.currentTimeMillis()}")
        val input = try {
            context.contentResolver.openInputStream(uri)
        } catch (e: Exception) {
            throw ImageLoadException("Couldn't open the selected image (no permission to read it).", e)
        } ?: throw ImageLoadException("Couldn't open the selected image (the gallery returned no data).")
        try {
            input.use { i -> FileOutputStream(out).use { o -> i.copyTo(o, 256 * 1024) } }
        } catch (e: Exception) {
            out.delete()
            throw ImageLoadException("Couldn't read the selected image.", e)
        }
        if (out.length() == 0L) {
            out.delete()
            throw ImageLoadException("The selected image is empty (0 bytes).")
        }
        Log.i(TAG, "copied $uri -> ${out.name} (${out.length()} bytes)")
        return out
    }

    /** Decode a local image file (JPEG/PNG/WebP/HEIC/AVIF/...) to an ARGB_8888 bitmap, longest side <= [maxSide]. */
    fun loadFile(file: File, maxSide: Int = MAX_SIDE): Bitmap {
        var first: Throwable? = null
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            try {
                return fitWithin(decodeWithImageDecoder(file, maxSide), maxSide)
            } catch (e: Throwable) {
                Log.w(TAG, "ImageDecoder failed for ${file.name}, falling back to BitmapFactory", e)
                first = e
            }
        }
        try {
            return fitWithin(decodeWithBitmapFactory(file, maxSide), maxSide)
        } catch (e: Throwable) {
            Log.e(TAG, "BitmapFactory failed for ${file.name}", e)
            val msg = if (e is OutOfMemoryError || first is OutOfMemoryError)
                "The selected image is too large to open on this phone."
            else "Couldn't decode the selected image (unsupported format or damaged file)."
            throw ImageLoadException(msg, first ?: e).also { if (first != null) it.addSuppressed(e) }
        }
    }

    /** Power-of-two subsample so that the decoded longest side stays >= maxSide (then scaled exactly). */
    private fun sampleFor(w: Int, h: Int, maxSide: Int): Int {
        var sample = 1
        while (max(w, h) / (sample * 2) >= maxSide) sample *= 2
        return sample
    }

    /** API 28+: handles HEIC/HEIF, AVIF (API 31+), WebP, PNG, JPEG; applies EXIF orientation itself. */
    @RequiresApi(Build.VERSION_CODES.P)
    private fun decodeWithImageDecoder(file: File, maxSide: Int): Bitmap {
        val source = ImageDecoder.createSource(file)
        val bmp = ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE   // pixels must be readable by OpenCV/MediaPipe
            decoder.isMutableRequired = true
            decoder.setTargetColorSpace(ColorSpace.get(ColorSpace.Named.SRGB))
            decoder.setOnPartialImageListener { true }             // show what we can of a truncated file
            decoder.setTargetSampleSize(sampleFor(info.size.width, info.size.height, maxSide))
        }
        return ensureArgb8888(bmp)
    }

    /** Fallback (API 24-27, or formats ImageDecoder rejects): BitmapFactory + ExifInterface. */
    private fun decodeWithBitmapFactory(file: File, maxSide: Int): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.path, bounds)   // always returns null in bounds mode - that's expected
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw IOException("BitmapFactory can't read this format (mime=${bounds.outMimeType})")
        }
        var sample = sampleFor(bounds.outWidth, bounds.outHeight, maxSide)
        var decoded: Bitmap? = null
        var attempts = 0
        while (decoded == null) {
            val opts = BitmapFactory.Options().apply {
                inSampleSize = sample
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            try {
                decoded = BitmapFactory.decodeFile(file.path, opts)
                    ?: throw IOException("BitmapFactory returned null (mime=${bounds.outMimeType}, ${bounds.outWidth}x${bounds.outHeight})")
            } catch (e: OutOfMemoryError) {
                if (++attempts >= 3) throw e
                sample *= 2
            }
        }
        val orientation = try {
            ExifInterface(file).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        } catch (e: Exception) {
            ExifInterface.ORIENTATION_NORMAL
        }
        val m = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.setScale(-1f, 1f)
            ExifInterface.ORIENTATION_ROTATE_180 -> m.setRotate(180f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> { m.setRotate(180f); m.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSPOSE -> { m.setRotate(90f); m.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_ROTATE_90 -> m.setRotate(90f)
            ExifInterface.ORIENTATION_TRANSVERSE -> { m.setRotate(-90f); m.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_ROTATE_270 -> m.setRotate(-90f)
        }
        val bmp = decoded!!
        if (m.isIdentity) return ensureArgb8888(bmp)
        val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        if (rotated !== bmp) bmp.recycle()
        return ensureArgb8888(rotated)
    }

    /** Scale down so the longest side is <= maxSide. */
    private fun fitWithin(bmp: Bitmap, maxSide: Int): Bitmap {
        val longest = max(bmp.width, bmp.height)
        if (longest <= maxSide) return bmp
        val s = maxSide.toFloat() / longest
        val scaled = Bitmap.createScaledBitmap(bmp, max(1, (bmp.width * s).roundToInt()), max(1, (bmp.height * s).roundToInt()), true)
        if (scaled !== bmp) bmp.recycle()
        return ensureArgb8888(scaled)
    }

    private fun ensureArgb8888(bmp: Bitmap): Bitmap {
        if (bmp.config == Bitmap.Config.ARGB_8888) return bmp
        val copy = bmp.copy(Bitmap.Config.ARGB_8888, true)
            ?: throw IOException("Couldn't convert image from ${bmp.config} to ARGB_8888")
        bmp.recycle()
        return copy
    }

    /** "ClassName: message <- CauseClass: message" for the small-print diagnostic line. */
    fun describe(e: Throwable): String {
        val parts = ArrayList<String>()
        var t: Throwable? = e
        while (t != null && parts.size < 4) {
            parts += "${t.javaClass.name}: ${t.message ?: ""}".trim()
            t = t.cause
        }
        e.suppressed.firstOrNull()?.let { parts += "also: ${it.javaClass.name}: ${it.message ?: ""}" }
        return parts.joinToString("  <-  ")
    }

    private fun fileName() =
        "FaceSwap_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".jpg"

    /** Save to Pictures/FaceSwap. Returns a human-readable location. */
    fun saveToGallery(context: Context, bmp: Bitmap): String {
        val name = fileName()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, name)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/FaceSwap")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
            val cr = context.contentResolver
            val uri = cr.insert(MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), values)
                ?: throw IOException("Couldn't create the image in the gallery.")
            try {
                cr.openOutputStream(uri)?.use { bmp.compress(Bitmap.CompressFormat.JPEG, 95, it) }
                    ?: throw IOException("Couldn't write the image.")
                values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0)
                cr.update(uri, values, null, null)
            } catch (e: Exception) {
                cr.delete(uri, null, null); throw e
            }
        } else {
            @Suppress("DEPRECATION")
            val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "FaceSwap")
            if (!dir.exists() && !dir.mkdirs()) throw IOException("Couldn't create Pictures/FaceSwap.")
            val f = File(dir, name)
            FileOutputStream(f).use { bmp.compress(Bitmap.CompressFormat.JPEG, 95, it) }
            MediaScannerConnection.scanFile(context, arrayOf(f.absolutePath), arrayOf("image/jpeg"), null)
        }
        return "Pictures/FaceSwap/$name"
    }

    /** Write to cache and build a share intent via FileProvider. */
    fun shareIntent(context: Context, bmp: Bitmap): Intent {
        val dir = File(context.cacheDir, "shared").apply { mkdirs() }
        val f = File(dir, "faceswap.jpg")
        FileOutputStream(f).use { bmp.compress(Bitmap.CompressFormat.JPEG, 92, it) }
        val uri = FileProvider.getUriForFile(context, context.packageName + ".fileprovider", f)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "image/jpeg"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        return Intent.createChooser(send, "Share face swap")
    }

    /** Uri for the camera app to write a capture into. */
    fun cameraUri(context: Context, slot: String): Uri {
        val dir = File(context.cacheDir, "camera").apply { mkdirs() }
        val f = File(dir, "capture_${slot}_${System.currentTimeMillis()}.jpg")
        return FileProvider.getUriForFile(context, context.packageName + ".fileprovider", f)
    }
}
