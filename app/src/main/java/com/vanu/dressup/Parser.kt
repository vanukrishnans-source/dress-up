package com.vanu.dressup

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

private fun mapModel(f: File) = RandomAccessFile(f, "r").use { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, it.length()) }

private fun floats(img: MPImage): FloatArray {
    val bb = ByteBufferExtractor.extract(img).order(ByteOrder.nativeOrder())
    bb.rewind()
    val fb = bb.asFloatBuffer(); return FloatArray(fb.remaining()).also { fb.get(it) }
}

/** Bilinear resize of a single-channel float map (align corners = false). */
fun resizeMap(src: FloatArray, sw: Int, sh: Int, dw: Int, dh: Int): FloatArray {
    if (sw == dw && sh == dh) return src
    val out = FloatArray(dw * dh)
    val fx = sw.toDouble() / dw; val fy = sh.toDouble() / dh
    for (y in 0 until dh) {
        val sy = ((y + 0.5) * fy - 0.5).coerceIn(0.0, sh - 1.0); val y0 = sy.toInt(); val y1 = min(y0 + 1, sh - 1); val wy = (sy - y0).toFloat()
        for (x in 0 until dw) {
            val sx = ((x + 0.5) * fx - 0.5).coerceIn(0.0, sw - 1.0); val x0 = sx.toInt(); val x1 = min(x0 + 1, sw - 1); val wx = (sx - x0).toFloat()
            val a = src[y0 * sw + x0] * (1 - wx) + src[y0 * sw + x1] * wx
            val b = src[y1 * sw + x0] * (1 - wx) + src[y1 * sw + x1] * wx
            out[y * dw + x] = a * (1 - wy) + b * wy
        }
    }
    return out
}

/**
 * MediaPipe parts (selfie multiclass: background, hair, body skin, face skin, clothes, others) + pose for one photo.
 * The photo is letterboxed to a grey square first so the 256x256 parts model never sees a squashed person
 * (tall full-body photos otherwise come out as "background").
 */
class PhotoParser(context: Context, segModel: File, poseModel: File) : Closeable {
    private val seg: ImageSegmenter = ImageSegmenter.createFromOptions(context,
        ImageSegmenter.ImageSegmenterOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetBuffer(mapModel(segModel)).setDelegate(Delegate.CPU).build())
            .setRunningMode(RunningMode.IMAGE).setOutputConfidenceMasks(true).setOutputCategoryMask(false).build())
    private val pose: PoseLandmarker = PoseLandmarker.createFromOptions(context,
        PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetBuffer(mapModel(poseModel)).setDelegate(Delegate.CPU).build())
            .setRunningMode(RunningMode.IMAGE).setNumPoses(1)
            .setMinPoseDetectionConfidence(0.4f).setMinPosePresenceConfidence(0.4f).build())

    fun parse(bmp: Bitmap): Parse {
        val W = bmp.width; val H = bmp.height
        // work copy at <= 1024 px (the models run at 256 anyway)
        val s = min(1.0, WORK / max(W, H).toDouble())
        val ww = max(1, (W * s).roundToInt()); val wh = max(1, (H * s).roundToInt())
        val small = if (s < 1.0) Bitmap.createScaledBitmap(bmp, ww, wh, true) else bmp
        val S = max(ww, wh); val ox = (S - ww) / 2; val oy = (S - wh) / 2
        val sq = Bitmap.createBitmap(S, S, Bitmap.Config.ARGB_8888)
        Canvas(sq).apply { drawColor(Color.rgb(127, 127, 127)); drawBitmap(small, null, Rect(ox, oy, ox + ww, oy + wh), Paint(Paint.FILTER_BITMAP_FLAG)) }
        val r = seg.segment(BitmapImageBuilder(sq).build())
        val ms = r.confidenceMasks().orElse(emptyList())
        fun cls(c: Int): FloatArray {
            if (c >= ms.size) return FloatArray(W * H)
            val img = ms[c]; val full = floats(img)
            val sqm = resizeMap(full, img.width, img.height, S, S)
            val crop = FloatArray(ww * wh)
            for (y in 0 until wh) System.arraycopy(sqm, (y + oy) * S + ox, crop, y * ww, ww)
            return resizeMap(crop, ww, wh, W, H)
        }
        val bg = cls(0); val hair = cls(1); val body = cls(2); val face = cls(3); val clothes = cls(4); val others = cls(5)
        sq.recycle()

        var shoulder: Double? = null; var hip: Double? = null; var faceVis = false
        val pr = pose.detect(BitmapImageBuilder(small).build())
        if (small !== bmp) small.recycle()
        val lms = pr.landmarks().firstOrNull()
        if (lms != null && lms.size > 24) {
            fun v(i: Int) = lms[i].visibility().orElse(0f)
            if (v(11) > 0.3f && v(12) > 0.3f) shoulder = (lms[11].y() + lms[12].y()) / 2.0 * H
            if (v(23) > 0.3f && v(24) > 0.3f) hip = (lms[23].y() + lms[24].y()) / 2.0 * H
            faceVis = v(0) > 0.5f && (v(2) > 0.5f || v(5) > 0.5f)
        }
        return Parse(W, H, hair, body, face, clothes, bg, others, shoulder, hip, faceVis)
    }

    override fun close() { seg.close(); pose.close() }

    companion object { const val WORK = 1024 }
}
