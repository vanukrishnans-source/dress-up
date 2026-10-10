package com.vanu.dressup

import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Desktop trial: runs the app's TryOnCore on parses written by reference/parse_photo.py. Set env TRIAL_DIR. */
class TrialRunner {
    private fun load(dir: File): Triple<IntArray, Int, Int> {
        val m = JSONObject(File(dir, "meta.json").readText()); val w = m.getInt("w"); val h = m.getInt("h")
        val b = File(dir, "rgb.raw").readBytes()
        return Triple(IntArray(w * h) { i -> (0xFF shl 24) or ((b[3 * i].toInt() and 255) shl 16) or ((b[3 * i + 1].toInt() and 255) shl 8) or (b[3 * i + 2].toInt() and 255) }, w, h)
    }

    private fun parse(dir: File): Parse {
        val m = JSONObject(File(dir, "meta.json").readText()); val w = m.getInt("w"); val h = m.getInt("h")
        val bb = ByteBuffer.wrap(File(dir, "maps.f32").readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val maps = Array(6) { FloatArray(w * h).also { a -> bb.get(a) } }
        fun d(k: String) = if (m.isNull(k)) null else m.getDouble(k)
        return Parse(w, h, maps[0], maps[1], maps[2], maps[3], maps[4], maps[5], d("shoulderY"), d("hipY"), m.getBoolean("faceVisible"))
    }

    private fun save(px: IntArray, w: Int, h: Int, f: File) {
        val b = ByteArray(w * h * 3); for (i in 0 until w * h) { b[3 * i] = (px[i] shr 16).toByte(); b[3 * i + 1] = (px[i] shr 8).toByte(); b[3 * i + 2] = px[i].toByte() }
        File(f.path + ".raw").writeBytes(b); File(f.path + ".size").writeText("$w $h")
    }

    @Test fun trial() {
        val root = System.getenv("TRIAL_DIR"); assumeTrue(root != null)
        val jobs = System.getenv("TRIAL_JOBS")!!.split(";")   // person,garment,KIND,outname
        for (j in jobs) {
            val (pn, gn, kind, out) = j.split(",")
            val pd = File(root, "parse/$pn"); val gd = File(root, "parse/$gn")
            val (p, pw, ph) = load(pd); val (g, gw, gh) = load(gd)
            val t0 = System.nanoTime()
            val r = TryOnCore.tryOn(p, pw, ph, parse(pd), g, gw, gh, parse(gd), GarmentKind.valueOf(kind))
            println("TRIAL $out ${pw}x$ph coverage=${"%.2f".format(r.coverage)} ${(System.nanoTime() - t0) / 1e6} ms ${r.note}")
            save(r.argb, pw, ph, File(root, "$out.png"))
            save(IntArray(pw * ph) { val v = (r.target[it] * 255).toInt().coerceIn(0, 255); (v shl 16) or (v shl 8) or v }, pw, ph, File(root, "debug/${out}_mask.png"))
            val gm = r.garmentMask!!
            save(IntArray(r.gw * r.gh) { val v = (gm[it] * 255).toInt().coerceIn(0, 255); (v shl 16) or (v shl 8) or v }, r.gw, r.gh, File(root, "debug/${out}_garment_mask.png"))
        }
    }
}
