package com.vanu.faceswap.core

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import java.io.ByteArrayInputStream
import kotlin.math.max

/**
 * Loads real image files through the same path the app uses for Photo Picker results
 * (content:// uri -> cache copy -> ImageDecoder on API 28+ / BitmapFactory+EXIF below).
 * Runs on API 33 (the reporting phone) and API 26 (BitmapFactory fallback path), with Robolectric's
 * native graphics so the real Android decoders are used.
 */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [26, 33])
class ImageUtilsTest {
    private val ctx: Context = ApplicationProvider.getApplicationContext()
    private var n = 0

    private fun bytes(name: String) = javaClass.getResourceAsStream("/images/$name")!!.readBytes()

    /** A Photo-Picker-style uri whose stream can be opened any number of times (fresh stream each time). */
    private fun pickerUri(name: String): Uri {
        val uri = Uri.parse("content://media/picker/0/com.android.providers.media.photopicker/media/${1000 + n++}")
        val data = bytes(name)
        shadowOf(ctx.contentResolver).registerInputStreamSupplier(uri) { ByteArrayInputStream(data) }
        return uri
    }

    private fun isRed(c: Int) = Color.red(c) > 200 && Color.green(c) < 60 && Color.blue(c) < 60

    private fun checkLandscape(b: Bitmap) {
        assertEquals(Bitmap.Config.ARGB_8888, b.config)
        assertEquals(300, b.width); assertEquals(200, b.height)
        assertTrue("red marker top-left", isRed(b.getPixel(5, 5)))
        assertTrue("blue elsewhere", !isRed(b.getPixel(250, 150)))
    }

    @Test fun jpegFromPhotoPicker() = checkLandscape(ImageUtils.load(ctx, pickerUri("plain.jpg"), "main"))
    @Test fun pngFromPhotoPicker() = checkLandscape(ImageUtils.load(ctx, pickerUri("plain.png"), "main"))
    @Test fun webpFromPhotoPicker() = checkLandscape(ImageUtils.load(ctx, pickerUri("plain.webp"), "faces"))

    @Test fun exifRotationApplied() {
        val b = ImageUtils.load(ctx, pickerUri("rot90.jpg"), "main")
        assertEquals(200, b.width); assertEquals(300, b.height)          // orientation 6 = rotate 90 CW
        assertTrue("marker moved to top-right", isRed(b.getPixel(b.width - 5, 5)))
    }

    @Test fun hugePhotoIsDownscaled() {
        val b = ImageUtils.load(ctx, pickerUri("huge.jpg"), "main")      // 9000x6000 (54 MP)
        assertEquals(ImageUtils.MAX_SIDE, max(b.width, b.height))
        assertEquals(1365, b.height)
        assertEquals(Bitmap.Config.ARGB_8888, b.config)
    }

    @Test fun corruptFileGivesClearError() = expectLoadError("corrupt.jpg", "decode")
    @Test fun nonImageGivesClearError() = expectLoadError("notimage.txt", "decode")

    @Test fun emptyStreamGivesClearError() {
        val uri = Uri.parse("content://media/picker/0/com.android.providers.media.photopicker/media/999")
        shadowOf(ctx.contentResolver).registerInputStreamSupplier(uri) { ByteArrayInputStream(ByteArray(0)) }
        try { ImageUtils.load(ctx, uri, "main"); fail("expected ImageLoadException") }
        catch (e: ImageLoadException) { assertTrue(e.message!!, e.message!!.contains("empty")) }
    }

    private fun expectLoadError(name: String, word: String) {
        try {
            ImageUtils.load(ctx, pickerUri(name), "main"); fail("expected ImageLoadException")
        } catch (e: ImageLoadException) {
            assertTrue(e.message!!, e.message!!.contains(word))
            assertTrue(ImageUtils.describe(e).contains("ImageLoadException"))
        }
    }
}

/**
 * HEIC: real Android 9+ devices decode it via ImageDecoder, but Robolectric's host graphics runtime has
 * no HEIF codec ('unimplemented'). So on the JVM we accept either a correct decode or a clean,
 * diagnosable ImageLoadException (never a crash / silent failure).
 */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [33])
class HeicTest {
    @Test fun heicFromPhotoPicker() {
        val ctx: Context = ApplicationProvider.getApplicationContext()
        val uri = Uri.parse("content://media/picker/0/com.android.providers.media.photopicker/media/5000")
        val data = javaClass.getResourceAsStream("/images/photo.heic")!!.readBytes()
        shadowOf(ctx.contentResolver).registerInputStreamSupplier(uri) { ByteArrayInputStream(data) }
        try {
            val b = ImageUtils.load(ctx, uri, "main")
            assertEquals(300, b.width); assertEquals(200, b.height)
        } catch (e: ImageLoadException) {
            val d = ImageUtils.describe(e)
            println("HEIC not decodable on this host (expected under Robolectric): $d")
            assertTrue(d, d.contains("ImageDecoder"))
        }
    }
}

/** Regression proof of the v1.0 bug: the old loader rejected every image, even a valid JPEG. */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [33])
class OldLoaderBugTest {
    @Test fun v10LoaderThrowsOnValidJpeg() {
        val ctx: Context = ApplicationProvider.getApplicationContext()
        val uri = Uri.parse("content://media/picker/0/com.android.providers.media.photopicker/media/7000")
        val data = javaClass.getResourceAsStream("/images/plain.jpg")!!.readBytes()
        shadowOf(ctx.contentResolver).registerInputStreamSupplier(uri) { ByteArrayInputStream(data) }
        // verbatim v1.0 code:
        val bounds = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }
        val result = ctx.contentResolver.openInputStream(uri)?.use { android.graphics.BitmapFactory.decodeStream(it, null, bounds) }
        assertEquals("bounds-only decode always returns null...", null, result)
        assertEquals("...even though the image is perfectly readable", 300, bounds.outWidth)
        // -> `?: throw IOException("Couldn't open that image.")` fired for every image in v1.0
        checkNotNull(ImageUtils.load(ctx, uri, "main"))   // v1.1 loads it
    }
}
