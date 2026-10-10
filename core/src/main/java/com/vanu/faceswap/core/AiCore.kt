package com.vanu.faceswap.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.io.EOFException
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/*
 * AI face-swap core (v2.0). Pure Kotlin + ONNX Runtime Java API, no Android classes, so the very same
 * file runs in the desktop parity harness. Mirrors test/ai_pipeline.py operation by operation:
 *
 *   5 keypoints from the MediaPipe mesh -> ArcFace w600k_r50 identity (112px crop)
 *   -> latent = emb @ emap / |emb| -> inswapper_128 on the 128px target crop
 *   -> paste back through the inverse similarity with a feathered box x face-oval mask
 *   -> GPEN-BFR-512 on the 512px FFHQ-aligned crop, blended 80 %, pasted with a feathered mask.
 *
 * Images are 8-bit RGB (the Python reference works in BGR; channel order is the only difference).
 */

/** 8-bit RGB image, interleaved (r,g,b per pixel). */
class RgbImage(val width: Int, val height: Int, val px: ByteArray = ByteArray(width * height * 3)) {
    fun copy() = RgbImage(width, height, px.copyOf())
}

/** Float image with [c] interleaved channels. */
class FImage(val width: Int, val height: Int, val c: Int, val data: FloatArray = FloatArray(width * height * c))

/** 468 MediaPipe face-mesh points, (x, y, z) per point, in image pixels. */
class FaceData(val pts: FloatArray) {
    fun x(i: Int) = pts[i * 3]
    fun y(i: Int) = pts[i * 3 + 1]
    val centerX: Float by lazy { var s = 0f; for (i in 0 until N) s += x(i); s / N }
    companion object { const val N = 468 }
}

class NoFaceException(message: String) : Exception(message)

object AiMath {
    private val ARC = doubleArrayOf(38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041)
    /** insightface arcface template, normalised to the 112px crop */
    val ARCFACE_112 = DoubleArray(10) { ARC[it] / 112.0 }
    /** insightface norm_crop(image_size=128): template shifted 8px right, normalised to 128 */
    val ARCFACE_128 = DoubleArray(10) { if (it % 2 == 0) (ARC[it] + 8.0) / 128.0 else (ARC[it] + 0.0) / 128.0 }
    val FFHQ_512 = doubleArrayOf(0.37691676, 0.46864664, 0.62285697, 0.46912813, 0.50123859, 0.61331904,
                                 0.39308822, 0.72541100, 0.61150205, 0.72490465)

    val EYE_A = intArrayOf(33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
    val EYE_B = intArrayOf(263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
    const val NOSE = 4
    val FACE_OVAL = intArrayOf(10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                               152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)

    /** [eye(image-left), eye(image-right), nose tip, mouth left, mouth right] as x0,y0,x1,y1,... */
    fun kps5(f: FaceData): DoubleArray {
        var ex1 = 0.0; var ey1 = 0.0; var ex2 = 0.0; var ey2 = 0.0
        for (i in EYE_A) { ex1 += f.x(i).toDouble(); ey1 += f.y(i).toDouble() }
        for (i in EYE_B) { ex2 += f.x(i).toDouble(); ey2 += f.y(i).toDouble() }
        return doubleArrayOf(ex1 / 16, ey1 / 16, ex2 / 16, ey2 / 16,
            f.x(NOSE).toDouble(), f.y(NOSE).toDouble(), f.x(61).toDouble(), f.y(61).toDouble(),
            f.x(291).toDouble(), f.y(291).toDouble())
    }

    /** Least-squares similarity src->dst (flat x,y arrays), returns 2x3 row-major. */
    fun umeyama(src: DoubleArray, dst: DoubleArray): DoubleArray {
        val n = src.size / 2
        var msx = 0.0; var msy = 0.0; var mdx = 0.0; var mdy = 0.0
        for (i in 0 until n) msx += src[2 * i]; msx /= n
        for (i in 0 until n) msy += src[2 * i + 1]; msy /= n
        for (i in 0 until n) mdx += dst[2 * i]; mdx /= n
        for (i in 0 until n) mdy += dst[2 * i + 1]; mdy /= n
        var a = 0.0; var b = 0.0; var v = 0.0
        for (i in 0 until n) {
            val sx = src[2 * i] - msx; val sy = src[2 * i + 1] - msy
            val dx = dst[2 * i] - mdx; val dy = dst[2 * i + 1] - mdy
            a += sx * dx + sy * dy; b += sx * dy - sy * dx; v += sx * sx + sy * sy
        }
        val ca = a / v; val sb = b / v
        return doubleArrayOf(ca, -sb, mdx - (ca * msx - sb * msy), sb, ca, mdy - (sb * msx + ca * msy))
    }

    fun invertAffine(m: DoubleArray): DoubleArray {
        val a = m[0]; val b = m[1]; val c = m[2]; val d = m[3]; val e = m[4]; val f = m[5]
        var det = a * e - b * d; det = if (det != 0.0) 1.0 / det else 0.0
        val a11 = e * det; val a22 = a * det; val a12 = -b * det; val a21 = -d * det
        return doubleArrayOf(a11, a12, -a11 * c - a12 * f, a21, a22, -a21 * c - a22 * f)
    }

    /** out(x,y) = bilinear sample of an 8-bit RGB image at A·(x,y,1), border replicate. */
    fun sample(src: RgbImage, a: DoubleArray, w: Int, h: Int): FImage {
        val out = FImage(w, h, 3); val o = out.data; val p = src.px
        val sw = src.width; val sh = src.height
        for (y in 0 until h) for (x in 0 until w) {
            val sx = a[0] * x + a[1] * y + a[2]; val sy = a[3] * x + a[4] * y + a[5]
            val flx = floor(sx); val fly = floor(sy); val fx = sx - flx; val fy = sy - fly
            val ix = flx.toInt(); val iy = fly.toInt()
            val x0 = ix.coerceIn(0, sw - 1); val x1 = (ix + 1).coerceIn(0, sw - 1)
            val y0 = iy.coerceIn(0, sh - 1); val y1 = (iy + 1).coerceIn(0, sh - 1)
            val i00 = (y0 * sw + x0) * 3; val i10 = (y0 * sw + x1) * 3; val i01 = (y1 * sw + x0) * 3; val i11 = (y1 * sw + x1) * 3
            val oi = (y * w + x) * 3
            for (ch in 0 until 3) {
                val p00 = (p[i00 + ch].toInt() and 255).toDouble(); val p10 = (p[i10 + ch].toInt() and 255).toDouble()
                val p01 = (p[i01 + ch].toInt() and 255).toDouble(); val p11 = (p[i11 + ch].toInt() and 255).toDouble()
                o[oi + ch] = ((1 - fy) * ((1 - fx) * p00 + fx * p10) + fy * ((1 - fx) * p01 + fx * p11)).toFloat()
            }
        }
        return out
    }

    /** Bilinear sample of a float image at A·(x,y,1); border replicate or zero. */
    fun sample(src: FImage, a: DoubleArray, w: Int, h: Int, replicate: Boolean): FImage {
        val c = src.c; val out = FImage(w, h, c); val o = out.data; val p = src.data
        val sw = src.width; val sh = src.height
        for (y in 0 until h) for (x in 0 until w) {
            val sx = a[0] * x + a[1] * y + a[2]; val sy = a[3] * x + a[4] * y + a[5]
            val flx = floor(sx); val fly = floor(sy); val fx = sx - flx; val fy = sy - fly
            val ix = flx.toInt(); val iy = fly.toInt()
            val oi = (y * w + x) * c
            if (replicate) {
                val x0 = ix.coerceIn(0, sw - 1); val x1 = (ix + 1).coerceIn(0, sw - 1)
                val y0 = iy.coerceIn(0, sh - 1); val y1 = (iy + 1).coerceIn(0, sh - 1)
                for (ch in 0 until c) {
                    val p00 = p[(y0 * sw + x0) * c + ch].toDouble(); val p10 = p[(y0 * sw + x1) * c + ch].toDouble()
                    val p01 = p[(y1 * sw + x0) * c + ch].toDouble(); val p11 = p[(y1 * sw + x1) * c + ch].toDouble()
                    o[oi + ch] = ((1 - fy) * ((1 - fx) * p00 + fx * p10) + fy * ((1 - fx) * p01 + fx * p11)).toFloat()
                }
            } else {
                val okx0 = ix in 0 until sw; val okx1 = ix + 1 in 0 until sw
                val oky0 = iy in 0 until sh; val oky1 = iy + 1 in 0 until sh
                for (ch in 0 until c) {
                    val p00 = if (okx0 && oky0) p[(iy * sw + ix) * c + ch].toDouble() else 0.0
                    val p10 = if (okx1 && oky0) p[(iy * sw + ix + 1) * c + ch].toDouble() else 0.0
                    val p01 = if (okx0 && oky1) p[((iy + 1) * sw + ix) * c + ch].toDouble() else 0.0
                    val p11 = if (okx1 && oky1) p[((iy + 1) * sw + ix + 1) * c + ch].toDouble() else 0.0
                    o[oi + ch] = ((1 - fy) * ((1 - fx) * p00 + fx * p10) + fy * ((1 - fx) * p01 + fx * p11)).toFloat()
                }
            }
        }
        return out
    }

    class Warp(val crop: FImage, val m: DoubleArray)

    /** Aligned crop: similarity kps -> template*size, sampled with border replicate. */
    fun warp(img: RgbImage, kps: DoubleArray, template: DoubleArray, size: Int): Warp {
        val dst = DoubleArray(10) { template[it] * size }
        val m = umeyama(kps, dst)
        return Warp(sample(img, invertAffine(m), size, size), m)
    }

    fun gaussKernel(sigma: Double): DoubleArray {
        val n = floor(sigma * 8 + 1 + 0.5).toInt() or 1
        val c = (n - 1) / 2.0
        val k = DoubleArray(n) { exp(-((it - c) * (it - c)) / (2 * sigma * sigma)) }
        val s = k.sum(); for (i in k.indices) k[i] /= s
        return k
    }

    private fun reflect101(i0: Int, n: Int): Int {
        if (n == 1) return 0
        var i = i0
        while (i < 0 || i >= n) { if (i < 0) i = -i; if (i >= n) i = 2 * n - 2 - i }
        return i
    }

    /** Separable Gaussian blur of a 1-channel image (BORDER_REFLECT_101). */
    fun gaussBlur(m: FImage, sigma: Double): FImage {
        val k = gaussKernel(sigma); val c = (k.size - 1) / 2; val w = m.width; val h = m.height
        val tmp = FloatArray(w * h); val out = FImage(w, h, 1); val src = m.data
        val xi = IntArray(w + 2 * c) { reflect101(it - c, w) }
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                var acc = 0.0
                for (j in k.indices) acc += k[j] * src[row + xi[x + j]].toDouble()
                tmp[row + x] = acc.toFloat()
            }
        }
        val yi = IntArray(h + 2 * c) { reflect101(it - c, h) }
        val o = out.data
        for (y in 0 until h) for (x in 0 until w) {
            var acc = 0.0
            for (j in k.indices) acc += k[j] * tmp[yi[y + j] * w + x].toDouble()
            o[y * w + x] = acc.toFloat()
        }
        return out
    }

    /** 1.0 where the pixel centre (x, y) is inside the polygon (crossing-number rule). */
    fun fillPolygon(size: Int, q: DoubleArray): FImage {
        val out = FImage(size, size, 1); val n = q.size / 2
        for (y in 0 until size) {
            val py = y.toDouble()
            for (x in 0 until size) {
                val px = x.toDouble(); var inside = false
                for (i in 0 until n) {
                    val j = if (i == 0) n - 1 else i - 1
                    val xi = q[2 * i]; val yi = q[2 * i + 1]; val xj = q[2 * j]; val yj = q[2 * j + 1]
                    if ((yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi) inside = !inside
                }
                if (inside) out.data[y * size + x] = 1f
            }
        }
        return out
    }

    private val boxCache = HashMap<Pair<Int, Double>, FImage>()

    /** Box of ones with zeroed borders, Gaussian-feathered (FaceFusion's static box mask). Cached. */
    fun boxMask(size: Int, blur: Double = 0.3): FImage = synchronized(boxCache) {
        boxCache.getOrPut(size to blur) {
            val amount = (size * 0.5 * blur).toInt(); val area = max(amount / 2, 1)
            val m = FImage(size, size, 1)
            for (y in 0 until size) for (x in 0 until size)
                if (y >= area && y < size - area && x >= area && x < size - area) m.data[y * size + x] = 1f
            if (amount > 0) gaussBlur(m, amount * 0.25) else m
        }
    }

    /** Target face-oval in crop space; upper half pushed up by 8 % of face height, grown by [grow]. */
    fun hullPolygon(f: FaceData, m: DoubleArray, grow: Double): DoubleArray {
        val n = FACE_OVAL.size
        val ox = DoubleArray(n) { f.x(FACE_OVAL[it]).toDouble() }; val oy = DoubleArray(n) { f.y(FACE_OVAL[it]).toDouble() }
        var cx = 0.0; var cy = 0.0
        for (i in 0 until n) cx += ox[i]; cx /= n
        for (i in 0 until n) cy += oy[i]; cy /= n
        var ux = f.x(10).toDouble() - f.x(152).toDouble(); var uy = f.y(10).toDouble() - f.y(152).toDouble()
        val fh = sqrt(ux * ux + uy * uy); ux /= (fh + 1e-6); uy /= (fh + 1e-6)
        val t = DoubleArray(n) { (ox[it] - cx) * ux + (oy[it] - cy) * uy }
        var tmax = 0.0; for (v in t) tmax = max(tmax, abs(v))
        val q = DoubleArray(2 * n)
        for (i in 0 until n) {
            val push = max(t[i], 0.0) / (tmax + 1e-6) * 0.08 * fh
            var x = ox[i] + push * ux; var y = oy[i] + push * uy
            x = cx + (x - cx) * (1 + grow); y = cy + (y - cy) * (1 + grow)
            q[2 * i] = m[0] * x + m[1] * y + m[2]; q[2 * i + 1] = m[3] * x + m[4] * y + m[5]
        }
        return q
    }

    fun hullMask(f: FaceData, m: DoubleArray, size: Int, grow: Double, feather: Double): FImage =
        gaussBlur(fillPolygon(size, hullPolygon(f, m, grow)), feather * size)

    fun mul(a: FImage, b: FImage): FImage = FImage(a.width, a.height, 1, FloatArray(a.data.size) { a.data[it] * b.data[it] })

    fun pasteBbox(m: DoubleArray, s: Int, w: Int, h: Int): IntArray {
        val mi = invertAffine(m)
        val cs = doubleArrayOf(0.0, 0.0, s.toDouble(), 0.0, 0.0, s.toDouble(), s.toDouble(), s.toDouble())
        var mnx = Double.MAX_VALUE; var mny = Double.MAX_VALUE; var mxx = -Double.MAX_VALUE; var mxy = -Double.MAX_VALUE
        for (i in 0 until 4) {
            val cx = cs[2 * i]; val cy = cs[2 * i + 1]
            val x = mi[0] * cx + mi[1] * cy + mi[2]; val y = mi[3] * cx + mi[4] * cy + mi[5]
            mnx = min(mnx, x); mxx = max(mxx, x); mny = min(mny, y); mxy = max(mxy, y)
        }
        return intArrayOf(max(floor(mnx).toInt() - 2, 0), max(floor(mny).toInt() - 2, 0),
                          min(ceil(mxx).toInt() + 2, w), min(ceil(mxy).toInt() + 2, h))
    }

    /** Blend [crop] (float RGB 0..255, crop space) into [frame] in place through the inverse of [m]. */
    fun paste(frame: RgbImage, crop: FImage, mask: FImage, m: DoubleArray) {
        val bb = pasteBbox(m, crop.width, frame.width, frame.height)
        val x0 = bb[0]; val y0 = bb[1]; val bw = bb[2] - x0; val bh = bb[3] - y0
        if (bw <= 0 || bh <= 0) return
        val a = doubleArrayOf(m[0], m[1], m[0] * x0 + m[1] * y0 + m[2], m[3], m[4], m[3] * x0 + m[4] * y0 + m[5])
        val inv = sample(crop, a, bw, bh, true)
        val im = sample(mask, a, bw, bh, false)
        val p = frame.px
        for (y in 0 until bh) for (x in 0 until bw) {
            val mk = im.data[y * bw + x].coerceIn(0f, 1f)
            val fi = ((y0 + y) * frame.width + x0 + x) * 3; val ci = (y * bw + x) * 3
            for (ch in 0 until 3) {
                val roi = (p[fi + ch].toInt() and 255).toFloat()
                val v = mk * inv.data[ci + ch] + (1f - mk) * roi + 0.5f
                p[fi + ch] = min(max(v, 0f), 255f).toInt().toByte()
            }
        }
    }
}

/**
 * Reads the last graph initializer of an ONNX file (inswapper's 512x512 "emap") by walking the
 * protobuf structure and seeking over everything else — no protobuf library, no full read.
 */
object OnnxInitializerReader {
    private class Reader(val f: RandomAccessFile) {
        fun varint(): Long {
            var shift = 0; var r = 0L
            while (true) {
                val b = f.read(); if (b < 0) throw EOFException()
                r = r or ((b and 0x7f).toLong() shl shift)
                if (b and 0x80 == 0) return r
                shift += 7
            }
        }
        fun skip(wt: Int) {
            when (wt) {
                0 -> varint()
                1 -> f.seek(f.filePointer + 8)
                2 -> { val n = varint(); f.seek(f.filePointer + n) }
                5 -> f.seek(f.filePointer + 4)
                else -> throw IllegalStateException("bad protobuf wire type $wt")
            }
        }
    }

    class Tensor(val name: String, val dims: LongArray, val data: FloatArray)

    fun readLast(file: File): Tensor = RandomAccessFile(file, "r").use { raf ->
        val r = Reader(raf); val len = raf.length()
        var last = -1L; var lastLen = 0L
        while (raf.filePointer < len) {
            val tag = r.varint(); val field = (tag ushr 3).toInt(); val wt = (tag and 7).toInt()
            if (field == 7 && wt == 2) {               // ModelProto.graph
                val glen = r.varint(); val gend = raf.filePointer + glen
                while (raf.filePointer < gend) {
                    val t = r.varint(); val fld = (t ushr 3).toInt(); val w = (t and 7).toInt()
                    if (fld == 5 && w == 2) {           // GraphProto.initializer
                        val n = r.varint(); last = raf.filePointer; lastLen = n; raf.seek(raf.filePointer + n)
                    } else r.skip(w)
                }
            } else r.skip(wt)
        }
        require(last >= 0) { "no initializer found in ${file.name}" }
        raf.seek(last); val end = last + lastLen
        var name = ""; val dims = ArrayList<Long>(); var dtype = 0; var raw: ByteArray? = null
        val floats = ArrayList<Float>()
        while (raf.filePointer < end) {
            val t = r.varint(); val fld = (t ushr 3).toInt(); val w = (t and 7).toInt()
            when {
                fld == 1 && w == 0 -> dims.add(r.varint())
                fld == 1 && w == 2 -> { val e = raf.filePointer + r.varint(); while (raf.filePointer < e) dims.add(r.varint()) }
                fld == 2 && w == 0 -> dtype = r.varint().toInt()
                fld == 8 && w == 2 -> { val b = ByteArray(r.varint().toInt()); raf.readFully(b); name = String(b) }
                fld == 9 && w == 2 -> { val b = ByteArray(r.varint().toInt()); raf.readFully(b); raw = b }
                fld == 4 && w == 2 -> {
                    val b = ByteArray(r.varint().toInt()); raf.readFully(b)
                    val fb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                    while (fb.hasRemaining()) floats.add(fb.get())
                }
                else -> r.skip(w)
            }
        }
        require(dtype == 1) { "initializer '$name' has data type $dtype, expected float32" }
        val data = raw?.let { b ->
            val fb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer(); FloatArray(fb.remaining()).also { fb.get(it) }
        } ?: floats.toFloatArray()
        Tensor(name, dims.toLongArray(), data)
    }
}

/** Paths of the three downloaded models. */
class AiModelFiles(val arcface: File, val swapper: File, val enhancer: File)

/**
 * Runs the pipeline with ONNX Runtime. Sessions are opened lazily per stage and closed right after
 * it (ArcFace -> swapper -> enhancer), so at most one big model is resident at a time.
 */
class AiEngine(
    private val files: AiModelFiles,
    private val threads: Int = 4,
    private val useXnnpack: Boolean = false,
    /** Video mode keeps sessions open for the whole job: without the CPU arena inswapper stays at
     *  ~0.43 GB instead of ~1.5 GB resident (same speed; GPEN-512 ~25% slower). */
    private val lowMemory: Boolean = false,
    private val log: (String) -> Unit = {},
) : Closeable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var emap: FloatArray? = null
    /** Which execution provider the last session actually used ("CPU" or "XNNPACK"). */
    var lastProvider = "CPU"; private set

    fun open(file: File): OrtSession {
        if (useXnnpack) {
            try {
                val o = OrtSession.SessionOptions()
                o.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR)
                o.setIntraOpNumThreads(1)                       // XNNPACK has its own thread pool
                if (lowMemory) o.setCPUArenaAllocator(false)
                o.addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                return env.createSession(file.absolutePath, o).also { lastProvider = "XNNPACK" }
            } catch (e: Throwable) {
                log("XNNPACK unavailable for ${file.name}, falling back to CPU: $e")
            }
        }
        val o = OrtSession.SessionOptions()
        o.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR)
        o.setIntraOpNumThreads(threads)
        if (lowMemory) o.setCPUArenaAllocator(false)
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        lastProvider = "CPU"
        return env.createSession(file.absolutePath, o)
    }

    private fun run1(s: OrtSession, feeds: Map<String, OnnxTensor>): FloatArray {
        try {
            s.run(feeds).use { r ->
                val t = r.get(0) as OnnxTensor
                val fb = t.floatBuffer; return FloatArray(fb.remaining()).also { fb.get(it) }
            }
        } finally { feeds.values.forEach { it.close() } }
    }

    private fun nchw(crop: FImage, f: (Float) -> Float): FloatBuffer {
        val n = crop.width * crop.height; val out = FloatArray(3 * n); val d = crop.data
        for (i in 0 until n) for (ch in 0 until 3) out[ch * n + i] = f(d[i * 3 + ch])
        return FloatBuffer.wrap(out)
    }

    private fun hwc(y: FloatArray, size: Int, f: (Float) -> Float): FImage {
        val n = size * size; val out = FImage(size, size, 3)
        for (i in 0 until n) for (ch in 0 until 3) out.data[i * 3 + ch] = f(y[ch * n + i])
        return out
    }

    /** Raw 512-d ArcFace embedding of the face at [kps] (plus the 112px crop, for tests). */
    fun embedding(s: OrtSession, img: RgbImage, kps: DoubleArray): Pair<FloatArray, FImage> {
        val w = AiMath.warp(img, kps, AiMath.ARCFACE_112, 112)
        val x = nchw(w.crop) { (it - 127.5f) / 127.5f }
        val t = OnnxTensor.createTensor(env, x, longArrayOf(1, 3, 112, 112))
        return run1(s, mapOf(s.inputNames.first() to t)) to w.crop
    }

    fun emap(): FloatArray = emap ?: OnnxInitializerReader.readLast(files.swapper).also {
        require(it.dims.contentEquals(longArrayOf(512, 512))) { "unexpected emap shape ${it.dims.toList()}" }
    }.data.also { emap = it }

    fun latent(emb: FloatArray): FloatArray {
        val e = emap(); var ss = 0.0
        for (v in emb) ss += v.toDouble() * v.toDouble()
        val norm = sqrt(ss)
        return FloatArray(512) { j ->
            var acc = 0.0
            for (i in 0 until 512) acc += emb[i].toDouble() * e[i * 512 + j].toDouble()
            (acc / norm).toFloat()
        }
    }

    class SwapDebug(val kps: DoubleArray, val m: DoubleArray, val crop: FImage, val out: FImage, val mask: FImage)

    /** inswapper forward pass: 128px RGB crop (0..255) + latent -> swapped RGB crop (0..255). */
    fun swapperForward(s: OrtSession, crop: FImage, latent: FloatArray): FImage {
        val x = OnnxTensor.createTensor(env, nchw(crop) { it / 255f }, longArrayOf(1, 3, 128, 128))
        val l = OnnxTensor.createTensor(env, FloatBuffer.wrap(latent), longArrayOf(1, 512))
        return hwc(run1(s, mapOf("target" to x, "source" to l)), 128) { it.coerceIn(0f, 1f) * 255f }
    }

    /** Swap one face in place. */
    fun swapFace(s: OrtSession, frame: RgbImage, target: FaceData, latent: FloatArray): SwapDebug {
        val kps = AiMath.kps5(target)
        val w = AiMath.warp(frame, kps, AiMath.ARCFACE_128, 128)
        val out = swapperForward(s, w.crop, latent)
        val mask = AiMath.mul(AiMath.boxMask(128, 0.3), AiMath.hullMask(target, w.m, 128, 0.04, 0.06))
        AiMath.paste(frame, out, mask, w.m)
        return SwapDebug(kps, w.m, w.crop, out, mask)
    }

    class EnhanceDebug(val m: DoubleArray, val crop: FImage, val out: FImage, val mask: FImage)

    /** GPEN forward pass: RGB crop (0..255) -> restored RGB crop (0..255), before blending. */
    fun enhancerForward(s: OrtSession, crop: FImage): FImage {
        val size = crop.width
        val x = OnnxTensor.createTensor(env, nchw(crop) { (it / 255f - 0.5f) / 0.5f }, longArrayOf(1, 3, size.toLong(), size.toLong()))
        return hwc(run1(s, mapOf(s.inputNames.first() to x)), size) { (it.coerceIn(-1f, 1f) + 1f) / 2f * 255f }
    }

    /** Restore detail on one face in place (GPEN-BFR-512, blended). */
    fun enhance(s: OrtSession, frame: RgbImage, target: FaceData, blend: Double = 0.8, size: Int = 512): EnhanceDebug {
        val w = AiMath.warp(frame, AiMath.kps5(target), AiMath.FFHQ_512, size)
        val out = enhancerForward(s, w.crop)
        val cb = (1.0 - blend).toFloat(); val bb = blend.toFloat()
        for (i in out.data.indices) out.data[i] = w.crop.data[i] * cb + out.data[i] * bb
        val mask = AiMath.mul(AiMath.boxMask(size, 0.3), AiMath.hullMask(target, w.m, size, 0.06, 0.05))
        AiMath.paste(frame, out, mask, w.m)
        return EnhanceDebug(w.m, w.crop, out, mask)
    }

    /**
     * Full pipeline: target face i (left->right) receives source face (i + rotation) mod sourceCount.
     * [progress] gets a user-facing stage label and a 0..1 fraction.
     */
    fun run(
        target: RgbImage, targetFaces: List<FaceData>, source: RgbImage, sourceFaces: List<FaceData>,
        rotation: Int, enhance: Boolean, progress: (String, Float) -> Unit = { _, _ -> },
    ): RgbImage {
        val n = min(targetFaces.size, sourceFaces.size)
        val wId = 0.06f; val wSwap = if (enhance) 0.34f else 0.94f; val wEnh = if (enhance) 0.6f else 0f
        val srcIdx = IntArray(n) { if (sourceFaces.size >= 2) Math.floorMod(it + rotation, sourceFaces.size) else 0 }
        progress("Reading faces…", 0f)
        val lat = open(files.arcface).use { s ->
            Array(n) { i -> latent(embedding(s, source, AiMath.kps5(sourceFaces[srcIdx[i]])).first) }
        }
        val out = target.copy()
        progress("Loading face-swap model…", wId)
        open(files.swapper).use { s ->
            for (i in 0 until n) {
                progress("Swapping face ${i + 1} of $n…", wId + wSwap * i / n)
                swapFace(s, out, targetFaces[i], lat[i])
            }
        }
        if (enhance) {
            progress("Enhancing detail…", wId + wSwap)
            open(files.enhancer).use { s ->
                for (i in 0 until n) {
                    progress(if (n > 1) "Enhancing detail (face ${i + 1} of $n)…" else "Enhancing detail…", wId + wSwap + wEnh * i / n)
                    enhance(s, out, targetFaces[i])
                }
            }
        }
        progress("Finishing…", 1f)
        return out
    }

    override fun close() { emap = null }
}
