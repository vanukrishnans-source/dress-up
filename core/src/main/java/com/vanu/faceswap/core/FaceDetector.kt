package com.vanu.faceswap.core

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import java.io.Closeable

/** MediaPipe FaceLandmarker wrapper (same detection + tiled rescans as v1.1 and the Python reference). */
class FaceDetector(context: Context, maxFaces: Int = MAX_FACES) : Closeable {

    private val landmarker: FaceLandmarker

    init {
        val base = BaseOptions.builder()
            .setModelAssetPath("face_landmarker.task")
            .setDelegate(Delegate.CPU)
            .build()
        val opts = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.IMAGE)
            .setNumFaces(maxFaces)
            .setMinFaceDetectionConfidence(0.4f)
            .setMinFacePresenceConfidence(0.4f)
            .build()
        landmarker = FaceLandmarker.createFromOptions(context, opts)
    }

    /**
     * Detect faces, sorted left-to-right by face centre x. The detector runs on a 128px input, so
     * when fewer than 2 faces are found (e.g. full-body shots with small faces) overlapping tiles
     * (2x2, then 3x3) are scanned too and new faces merged.
     */
    fun detect(bitmap: Bitmap): List<FaceData> {
        val faces = detectRegion(bitmap, 0, 0, bitmap.width, bitmap.height).toMutableList()
        if (faces.size < 2) {
            val w = bitmap.width; val h = bitmap.height
            for (grid in intArrayOf(2, 3)) {
                val tw = (w / grid.toFloat() * 1.5f).toInt().coerceAtMost(w)
                val th = (h / grid.toFloat() * 1.5f).toInt().coerceAtMost(h)
                for (gy in 0 until grid) for (gx in 0 until grid) {
                    val x0 = Math.round((w - tw) * gx / (grid - 1).toFloat())
                    val y0 = Math.round((h - th) * gy / (grid - 1).toFloat())
                    for (f in detectRegion(bitmap, x0, y0, tw, th)) mergeFace(faces, f)
                }
                if (faces.size >= 2) break
            }
        }
        return faces.sortedBy { it.centerX }
    }

    /** Detect faces inside a sub-rectangle; points are returned in full-bitmap coordinates. */
    fun detectRegion(bitmap: Bitmap, x0: Int, y0: Int, rw: Int, rh: Int): List<FaceData> {
        val crop = if (x0 == 0 && y0 == 0 && rw == bitmap.width && rh == bitmap.height) bitmap
                   else Bitmap.createBitmap(bitmap, x0, y0, rw, rh)
        try {
            val result = landmarker.detect(BitmapImageBuilder(crop).build())
            val w = crop.width.toFloat(); val h = crop.height.toFloat()
            return result.faceLandmarks().filter { it.size >= FaceData.N }.map { lms ->
                val a = FloatArray(FaceData.N * 3)
                for (i in 0 until FaceData.N) {
                    val l = lms[i]
                    a[i * 3] = l.x() * w + x0; a[i * 3 + 1] = l.y() * h + y0; a[i * 3 + 2] = l.z() * w
                }
                FaceData(a)
            }
        } finally {
            if (crop !== bitmap) crop.recycle()
        }
    }

    private fun width(f: FaceData): Float {
        var mn = Float.MAX_VALUE; var mx = -Float.MAX_VALUE
        for (i in 0 until FaceData.N) { mn = minOf(mn, f.x(i)); mx = maxOf(mx, f.x(i)) }
        return mx - mn
    }

    private fun centerY(f: FaceData): Float { var s = 0f; for (i in 0 until FaceData.N) s += f.y(i); return s / FaceData.N }

    /** Add [f] unless a face with a nearby centre is already in [faces]. */
    fun mergeFace(faces: MutableList<FaceData>, f: FaceData) {
        val fw = width(f)
        for (g in faces) {
            val dx = g.centerX - f.centerX; val dy = centerY(g) - centerY(f)
            if (kotlin.math.sqrt(dx * dx + dy * dy) < 0.5f * maxOf(fw, width(g))) return
        }
        faces.add(f)
    }

    override fun close() = landmarker.close()

    companion object {
        const val MAX_FACES = 6

        fun toRgb(bmp: Bitmap): RgbImage {
            val src = if (bmp.config == Bitmap.Config.ARGB_8888) bmp else bmp.copy(Bitmap.Config.ARGB_8888, false)
            val w = src.width; val h = src.height
            val out = RgbImage(w, h); val row = IntArray(w)
            for (y in 0 until h) {
                src.getPixels(row, 0, w, 0, y, w, 1)
                var o = y * w * 3
                for (x in 0 until w) {
                    val c = row[x]
                    out.px[o] = (c shr 16).toByte(); out.px[o + 1] = (c shr 8).toByte(); out.px[o + 2] = c.toByte(); o += 3
                }
            }
            if (src !== bmp) src.recycle()
            return out
        }

        fun toBitmap(img: RgbImage): Bitmap {
            val w = img.width; val h = img.height
            val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888); val row = IntArray(w)
            for (y in 0 until h) {
                var i = y * w * 3
                for (x in 0 until w) {
                    row[x] = (0xff shl 24) or ((img.px[i].toInt() and 255) shl 16) or
                             ((img.px[i + 1].toInt() and 255) shl 8) or (img.px[i + 2].toInt() and 255)
                    i += 3
                }
                bmp.setPixels(row, 0, w, 0, y, w, 1)
            }
            return bmp
        }
    }
}
