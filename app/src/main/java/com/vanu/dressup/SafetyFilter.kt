package com.vanu.dressup

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import com.vanu.faceswap.core.RgbImage
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer

class SafetyBlockedException(val what: String, val score: Float) : Exception("safety filter blocked the $what")

/**
 * Always-on NSFW filter (Falconsai/nsfw_image_detection ViT, int8): same model family, preprocessing
 * and thresholds as AI Image Create. Input: the whole image stretched to 224x224 (area-averaged),
 * normalised to [-1, 1]; output probs[1] = "nsfw".
 */
class SafetyFilter(model: File, threads: Int = 2) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = OrtSession.SessionOptions().use { o ->
        o.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR)
        o.setIntraOpNumThreads(threads)
        o.setCPUArenaAllocator(false)
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        env.createSession(model.absolutePath, o)
    }

    fun score(img: RgbImage): Float {
        val x = preprocess(img)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())).use { t ->
            session.run(mapOf("pixel_values" to t)).use { r ->
                // Falconsai id2label: 0=normal, 1=nsfw. Copy via remaining() like AI Image Create
                // so we never depend on absolute FloatBuffer indexing quirks across ORT builds.
                val fb = (r.get(0) as OnnxTensor).floatBuffer
                val probs = FloatArray(fb.remaining()).also { fb.get(it) }
                require(probs.size >= 2) { "safety model returned ${probs.size} probs, expected 2" }
                return probs[1]   // P(nsfw)
            }
        }
    }

    override fun close() = session.close()

    companion object {
        const val SIZE = 224

        /** Area-average resize to 224x224 (stretch), CHW, (v/255 - 0.5) / 0.5. */
        fun preprocess(img: RgbImage): FloatArray {
            val w = img.width; val h = img.height; val m = SIZE * SIZE
            val out = FloatArray(3 * m)
            val sx = w.toDouble() / SIZE; val sy = h.toDouble() / SIZE
            for (oy in 0 until SIZE) {
                val y0 = oy * sy; val y1 = (oy + 1) * sy
                for (ox in 0 until SIZE) {
                    val x0 = ox * sx; val x1 = (ox + 1) * sx
                    var r = 0.0; var g = 0.0; var b = 0.0; var wt = 0.0
                    var yy = kotlin.math.floor(y0).toInt()
                    while (yy < y1 && yy < h) {
                        val wy = minOf(y1, yy + 1.0) - maxOf(y0, yy.toDouble())
                        var xx = kotlin.math.floor(x0).toInt()
                        while (xx < x1 && xx < w) {
                            val wx = minOf(x1, xx + 1.0) - maxOf(x0, xx.toDouble())
                            val k = wx * wy; val i = (yy * w + xx) * 3
                            r += k * (img.px[i].toInt() and 255); g += k * (img.px[i + 1].toInt() and 255); b += k * (img.px[i + 2].toInt() and 255)
                            wt += k; xx++
                        }
                        yy++
                    }
                    val o = oy * SIZE + ox
                    out[o] = ((r / wt / 255.0 - 0.5) / 0.5).toFloat()
                    out[m + o] = ((g / wt / 255.0 - 0.5) / 0.5).toFloat()
                    out[2 * m + o] = ((b / wt / 255.0 - 0.5) / 0.5).toFloat()
                }
            }
            return out
        }
    }
}

/** Always on; only the threshold can be chosen. Default Relaxed (same as AI Image Create / Collage Video 1.0.1). */
enum class FilterStrictness(val threshold: Float, val label: String, val help: String) {
    RELAXED(0.85f, "Relaxed", "Blocks only clearly explicit content (score above 0.85)."),
    STANDARD(0.5f, "Standard", "Blocks anything that looks likely to be explicit (score above 0.5)."),
}
