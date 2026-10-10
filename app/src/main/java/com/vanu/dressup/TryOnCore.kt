package com.vanu.dressup

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/*
 * Dress Up try-on maths (pure Kotlin, no Android classes, so the same code runs in the app and in the
 * desktop trial on the box).
 *
 * Method (classical, on-device; no generative model):
 *  1. Person: MediaPipe multiclass parts (hair / body skin / face skin / clothes) + pose keypoints.
 *     The target is the person's CURRENT clothes region (upper / lower / full by the hip line), with
 *     skin, face and hair excluded. The person's silhouette, pose, face, hair, skin and background are
 *     never changed: only pixels inside that clothes region are replaced.
 *  2. Garment: cut out (product / mannequin photo: background colour from the border, largest blob,
 *     holes filled, stand + neck trimmed; worn photo: the wearer's clothes region), then the texture is
 *     extended past its edges (pull-push) so the warp never samples background.
 *  3. Warp: row-by-row span mapping (a 1-D thin-plate-like stretch): each row of the target region maps
 *     to the matching relative row of the garment, and the row's left..right extent to the garment's
 *     left..right extent. The garment therefore follows the body outline exactly.
 *  4. Shading: the original clothes' folds and light (luminance / local mean) are multiplied onto the
 *     new garment, then it is feathered in.
 */

enum class GarmentKind(val label: String) { UPPER("Top"), LOWER("Bottom"), FULL("Full dress") }

/** Parse of one photo: soft maps (0..1, w*h) and pose rows (y px), null when the pose wasn't found. */
class Parse(
    val w: Int, val h: Int,
    val hair: FloatArray, val skin: FloatArray, val face: FloatArray, val clothes: FloatArray, val background: FloatArray, val others: FloatArray,
    val shoulderY: Double?, val hipY: Double?, val faceVisible: Boolean,
)

class TryOnResult(val argb: IntArray, val w: Int, val h: Int, val target: FloatArray, val coverage: Float, val note: String?,
                  val garmentMask: FloatArray? = null, val gw: Int = 0, val gh: Int = 0)

class TryOnException(msg: String) : Exception(msg)

object TryOnCore {
    const val MIN_TARGET_FRACTION = 0.004f

    fun smoothstep(x: Float, a: Float, b: Float): Float { val t = ((x - a) / (b - a)).coerceIn(0f, 1f); return t * t * (3 - 2 * t) }

    // ------------------------------------------------------------------ fast blur (3 x box ~ Gaussian), O(n)
    fun boxBlur(src: FloatArray, w: Int, h: Int, radius: Int): FloatArray {
        if (radius < 1) return src.copyOf()
        val tmp = FloatArray(src.size); val out = FloatArray(src.size); val n = 2 * radius + 1
        for (y in 0 until h) {
            val o = y * w; var s = 0f
            for (i in -radius..radius) s += src[o + i.coerceIn(0, w - 1)]
            for (x in 0 until w) {
                tmp[o + x] = s / n
                s += src[o + (x + radius + 1).coerceAtMost(w - 1)] - src[o + (x - radius).coerceAtLeast(0)]
            }
        }
        for (x in 0 until w) {
            var s = 0f
            for (i in -radius..radius) s += tmp[i.coerceIn(0, h - 1) * w + x]
            for (y in 0 until h) {
                out[y * w + x] = s / n
                s += tmp[(y + radius + 1).coerceAtMost(h - 1) * w + x] - tmp[(y - radius).coerceAtLeast(0) * w + x]
            }
        }
        return out
    }

    fun blur(src: FloatArray, w: Int, h: Int, sigma: Double): FloatArray {
        if (sigma < 0.5) return src.copyOf()
        // three box passes with r ~ sigma * sqrt(3) / sqrt(3 passes)
        val r = max(1, (sigma * 1.0).roundToInt())
        return boxBlur(boxBlur(boxBlur(src, w, h, r), w, h, r), w, h, r)
    }

    // ------------------------------------------------------------------ target region on the person
    /**
     * Soft target (0..1) = the person's clothes, minus skin / face / hair, gated vertically by [kind]:
     * Top = above the hip line, Bottom = below it, Full = everything below the shoulders.
     */
    fun targetRegion(p: Parse, kind: GarmentKind): FloatArray {
        val w = p.w; val h = p.h
        // clothes rows (for fallbacks when the pose is missing)
        var cy0 = h; var cy1 = -1
        for (y in 0 until h) for (x in 0 until w) if (p.clothes[y * w + x] > 0.5f) { if (y < cy0) cy0 = y; if (y > cy1) cy1 = y }
        if (cy1 < 0) return FloatArray(w * h)
        val sh = p.shoulderY ?: (cy0 + 0.05 * (cy1 - cy0))
        val hip = p.hipY ?: (cy0 + 0.45 * (cy1 - cy0))
        val torso = max(8.0, hip - sh)
        val top = (sh - 0.04 * torso).toFloat()            // keeps head scarves / hats (above the shoulders) out of the change
        val band = (0.05 * torso).toFloat()
        val out = FloatArray(w * h)
        for (y in 0 until h) {
            val yf = y.toFloat()
            val gTop = smoothstep(yf, top - band, top + band)
            val g = when (kind) {
                GarmentKind.FULL -> gTop
                GarmentKind.UPPER -> gTop * (1f - smoothstep(yf, hip.toFloat() - band, hip.toFloat() + band))
                GarmentKind.LOWER -> smoothstep(yf, hip.toFloat() - 2 * band, hip.toFloat())
            }
            if (g <= 0f) continue
            for (x in 0 until w) {
                val i = y * w + x
                val c = p.clothes[i]
                val other = max(max(p.skin[i], 1.6f * p.others[i]), max(p.face[i], p.hair[i]))
                val v = smoothstep(c - other, -0.1f, 0.25f) * smoothstep(c, 0.25f, 0.55f)
                out[i] = v * g
            }
        }
        return out
    }

    // ------------------------------------------------------------------ garment cut-out
    /** 0..1: how different each pixel is from the border (backdrop) colour; 0 = looks like backdrop. */
    fun backdropDistance(argb: IntArray, w: Int, h: Int): FloatArray {
        val rs = ArrayList<Int>(); val gs = ArrayList<Int>(); val bs = ArrayList<Int>()
        val step = max(1, (w + h) / 400)
        fun add(x: Int, y: Int) { val c = argb[y * w + x]; rs.add((c shr 16) and 255); gs.add((c shr 8) and 255); bs.add(c and 255) }
        for (x in 0 until w step step) { add(x, 0); add(x, min(h - 1, 2)) }
        for (y in 0 until h step step) { add(0, y); add(w - 1, y) }
        fun med(l: ArrayList<Int>) = l.sorted()[l.size / 2]
        val br = med(rs); val bg = med(gs); val bb = med(bs)
        val d = rs.indices.map { val dr = rs[it] - br; val dg = gs[it] - bg; val db = bs[it] - bb; sqrt((dr * dr + dg * dg + db * db).toDouble()) }.sorted()
        val thr = (d[(d.size * 0.9).toInt().coerceAtMost(d.size - 1)] + 10.0).coerceIn(10.0, 45.0).toFloat()
        return FloatArray(w * h) { i -> val c = argb[i]; val dr = ((c shr 16) and 255) - br; val dg = ((c shr 8) and 255) - bg; val db = (c and 255) - bb
            smoothstep(sqrt((dr * dr + dg * dg + db * db).toFloat()), thr * 0.5f, thr) }
    }

    /** Product / mannequin photo: soft garment mask from a border background-colour model. */
    fun productMask(argb: IntArray, w: Int, h: Int): FloatArray {
        // border samples
        val rs = ArrayList<Int>(); val gs = ArrayList<Int>(); val bs = ArrayList<Int>()
        val step = max(1, (w + h) / 400)
        fun add(x: Int, y: Int) { val c = argb[y * w + x]; rs.add((c shr 16) and 255); gs.add((c shr 8) and 255); bs.add(c and 255) }
        for (x in 0 until w step step) { add(x, 0); add(x, h - 1); add(x, min(h - 1, 2)); add(x, max(0, h - 3)) }
        for (y in 0 until h step step) { add(0, y); add(w - 1, y); add(min(w - 1, 2), y); add(max(0, w - 3), y) }
        fun med(l: ArrayList<Int>) = l.sorted()[l.size / 2]
        val br = med(rs); val bg = med(gs); val bb = med(bs)
        val dists = rs.indices.map { val dr = rs[it] - br; val dg = gs[it] - bg; val db = bs[it] - bb; sqrt((dr * dr + dg * dg + db * db).toDouble()) }.sorted()
        val thr = (dists[(dists.size * 0.9).toInt().coerceAtMost(dists.size - 1)] * 1.25 + 12.0).coerceIn(16.0, 80.0)
        // background colour can drift (vignette): also accept smooth gradients via a blurred border-distance
        val fg = BooleanArray(w * h)
        for (i in 0 until w * h) {
            val c = argb[i]; val dr = ((c shr 16) and 255) - br; val dg = ((c shr 8) and 255) - bg; val db = (c and 255) - bb
            fg[i] = sqrt((dr * dr + dg * dg + db * db).toDouble()) > thr
        }
        // background = flood from the border through non-foreground pixels; everything else is the garment
        // (this fills holes such as a pattern matching the backdrop)
        // close small gaps (prints that match the backdrop) before picking the blob
        val fgF = FloatArray(w * h) { if (fg[it]) 1f else 0f }
        val closed = blur(fgF, w, h, max(2.0, min(w, h) / 150.0))
        val fg2 = BooleanArray(w * h) { closed[it] > 0.4f }
        var keep = largestComponent(fg2, w, h)
        keep = fillHoles(keep, w, h)
        // trim mannequin neck (top) and stand (bottom): rows narrower than 30% of the widest row at the ends
        val widths = IntArray(h)
        for (y in 0 until h) { var x0 = -1; var x1 = -1; for (x in 0 until w) if (keep[y * w + x]) { if (x0 < 0) x0 = x; x1 = x }; widths[y] = if (x0 < 0) 0 else x1 - x0 + 1 }
        val maxW = widths.maxOrNull() ?: 0
        if (maxW == 0) throw TryOnException("No garment found in the clothing photo. Use a photo with a plain background, or a photo of someone wearing it.")
        val lim = (maxW * 0.30).toInt()
        var y0 = 0; while (y0 < h && widths[y0] < lim) y0++
        var y1 = h - 1; while (y1 > y0 && widths[y1] < lim) y1--
        val m = FloatArray(w * h)
        for (y in y0..y1) for (x in 0 until w) if (keep[y * w + x]) m[y * w + x] = 1f
        return blur(m, w, h, max(1.0, min(w, h) / 600.0))
    }

    /** Connected components (4-neighbour) of [fg]: labels (0 = none) and, per label, size and edge contact. */
    private class Comps(val label: IntArray, val size: IntArray, val leftRight: BooleanArray, val count: Int)

    private fun components(fg: BooleanArray, w: Int, h: Int): Comps {
        val label = IntArray(w * h); val sizes = ArrayList<Int>(); val lr = ArrayList<Boolean>(); sizes.add(0); lr.add(false)
        val stack = IntArray(w * h); var cur = 0
        for (s in 0 until w * h) {
            if (!fg[s] || label[s] != 0) continue
            cur++; var sp = 0; stack[sp++] = s; label[s] = cur; var n = 0; var edge = false
            while (sp > 0) {
                val i = stack[--sp]; n++
                val x = i % w; val y = i / w
                if (x == 0 || x == w - 1) edge = true
                if (x > 0 && fg[i - 1] && label[i - 1] == 0) { label[i - 1] = cur; stack[sp++] = i - 1 }
                if (x < w - 1 && fg[i + 1] && label[i + 1] == 0) { label[i + 1] = cur; stack[sp++] = i + 1 }
                if (y > 0 && fg[i - w] && label[i - w] == 0) { label[i - w] = cur; stack[sp++] = i - w }
                if (y < h - 1 && fg[i + w] && label[i + w] == 0) { label[i + w] = cur; stack[sp++] = i + w }
            }
            sizes.add(n); lr.add(edge)
        }
        return Comps(label, sizes.toIntArray(), lr.toBooleanArray(), cur)
    }

    /** The garment blob: the biggest component, strongly preferring ones that don't touch the left / right edge (floor, walls). */
    private fun largestComponent(fg: BooleanArray, w: Int, h: Int): BooleanArray {
        val c = components(fg, w, h)
        var best = 0; var bestScore = 0.0
        for (k in 1..c.count) { val sc = c.size[k] * (if (c.leftRight[k]) 0.15 else 1.0); if (sc > bestScore) { bestScore = sc; best = k } }
        return BooleanArray(w * h) { best != 0 && c.label[it] == best }
    }

    /** Fills only small holes (print matching the backdrop); big enclosed gaps (between sleeve and body) stay open. */
    private fun fillHoles(m: BooleanArray, w: Int, h: Int): BooleanArray {
        val inv = BooleanArray(w * h) { !m[it] }
        val c = components(inv, w, h)
        val maxHole = (w * h * 0.002).toInt()
        val out = m.copyOf()
        // a hole doesn't touch any border
        val touches = BooleanArray(c.count + 1)
        for (x in 0 until w) { touches[c.label[x]] = true; touches[c.label[(h - 1) * w + x]] = true }
        for (y in 0 until h) { touches[c.label[y * w]] = true; touches[c.label[y * w + w - 1]] = true }
        for (i in 0 until w * h) { val l = c.label[i]; if (l != 0 && !touches[l] && c.size[l] <= maxHole) out[i] = true }
        return out
    }

    /** The garment mask for [kind]: worn photo = the wearer's clothes region; product = cut-out, then sliced at the waist. */
    fun garmentMask(argb: IntArray, w: Int, h: Int, parse: Parse?, kind: GarmentKind): Pair<FloatArray, Boolean> {
        if (parse != null && parse.faceVisible) {
            val t = targetRegion(parse, kind)
            var n = 0; for (v in t) if (v > 0.5f) n++
            if (n > w * h * MIN_TARGET_FRACTION) return t to true
        }
        // product / mannequin / flat lay: the parts model's "clothes" class works on garments without a person too;
        // fall back to a background-colour cut-out when it finds too little
        var m: FloatArray? = null
        if (parse != null) {
            val c = FloatArray(w * h) { i -> smoothstep(parse.clothes[i] - max(parse.background[i], parse.others[i]), -0.1f, 0.3f) }
            var n = 0; for (v in c) if (v > 0.5f) n++
            if (n > w * h * 0.02f) {
                val fg = BooleanArray(w * h) { c[it] > 0.5f }
                val keep = fillHoles(largestComponent(fg, w, h), w, h)
                // pixels that look like the backdrop: only big regions count (the gap between sleeve and body);
                // small specks inside a print that happen to match the backdrop stay garment
                val raw = backdropDistance(argb, w, h)
                val solid = fillHoles(BooleanArray(w * h) { raw[it] > 0.5f }, w, h)
                val bgd = FloatArray(w * h) { if (solid[it]) 1f else raw[it] }
                m = FloatArray(w * h) { if (keep[it]) c[it].coerceAtLeast(0.51f) * bgd[it] else c[it] * 0.4f * bgd[it] }
            }
        }
        if (m == null) m = productMask(argb, w, h)
        if (kind == GarmentKind.FULL) return m to false
        // waist = narrowest row between 20% and 65% of the garment height
        val rows = rowSpans(m, w, h, 0.5f)
        val ys = (0 until h).filter { rows[2 * it] >= 0 }
        if (ys.size < 8) return m to false
        val gy0 = ys.first(); val gy1 = ys.last(); val gh = gy1 - gy0
        var waist = gy0 + gh / 2; var best = Int.MAX_VALUE
        for (y in (gy0 + gh / 5)..(gy0 + gh * 13 / 20)) { val wd = rows[2 * y + 1] - rows[2 * y]; if (rows[2 * y] >= 0 && wd < best) { best = wd; waist = y } }
        // a plain top / bottom product photo has no real waist (narrowest row is nearly as wide as the rest): keep it whole
        val widest = ys.maxOf { rows[2 * it + 1] - rows[2 * it] }
        if (best > widest * 0.8) return m to false
        val out = m.copyOf(); val band = max(2, gh / 40)
        for (y in 0 until h) {
            val g = when (kind) {
                GarmentKind.UPPER -> 1f - smoothstep(y.toFloat(), (waist - band).toFloat(), (waist + band).toFloat())
                GarmentKind.LOWER -> smoothstep(y.toFloat(), (waist - band).toFloat(), (waist + band).toFloat())
                GarmentKind.FULL -> 1f
            }
            for (x in 0 until w) out[y * w + x] *= g
        }
        return out to false
    }

    /** Per-row [left, right] of mask > thr (−1 when empty), as an IntArray of 2*h. */
    fun rowSpans(m: FloatArray, w: Int, h: Int, thr: Float): IntArray {
        val r = IntArray(2 * h) { -1 }
        for (y in 0 until h) {
            var x0 = -1; var x1 = -1
            for (x in 0 until w) if (m[y * w + x] > thr) { if (x0 < 0) x0 = x; x1 = x }
            r[2 * y] = x0; r[2 * y + 1] = x1
        }
        return r
    }

    /** Smooth spans over rows (moving average over a window), filling empty rows inside [y0, y1] by interpolation. */
    fun smoothSpans(spans: IntArray, h: Int, win: Int): Pair<DoubleArray, DoubleArray>? {
        val ys = (0 until h).filter { spans[2 * it] >= 0 }
        if (ys.size < 4) return null
        val y0 = ys.first(); val y1 = ys.last()
        val L = DoubleArray(h) { Double.NaN }; val R = DoubleArray(h) { Double.NaN }
        var prev = y0
        for (y in y0..y1) {
            if (spans[2 * y] >= 0) { L[y] = spans[2 * y].toDouble(); R[y] = spans[2 * y + 1] + 1.0; prev = y }
        }
        // interpolate gaps
        var last = y0
        for (y in y0 + 1..y1) if (!L[y].isNaN()) {
            if (y - last > 1) for (k in last + 1 until y) { val t = (k - last).toDouble() / (y - last); L[k] = L[last] + t * (L[y] - L[last]); R[k] = R[last] + t * (R[y] - R[last]) }
            last = y
        }
        val hw = max(1, win / 2)
        val SL = DoubleArray(h) { Double.NaN }; val SR = DoubleArray(h) { Double.NaN }
        for (y in y0..y1) {
            var sl = 0.0; var sr = 0.0; var n = 0
            for (k in max(y0, y - hw)..min(y1, y + hw)) { sl += L[k]; sr += R[k]; n++ }
            SL[y] = sl / n; SR[y] = sr / n
        }
        @Suppress("UNUSED_VARIABLE") val unused = prev
        return SL to SR
    }

    // ------------------------------------------------------------------ texture extension (pull-push)
    /** Extends the garment colours outside its mask so bilinear samples near the edge never pick up the backdrop. */
    fun extendTexture(argb: IntArray, mask: FloatArray, w: Int, h: Int): Array<FloatArray> {
        val levels = ArrayList<Array<FloatArray>>(); val dims = ArrayList<IntArray>()
        var cw = w; var ch = h
        var cur = Array(4) { FloatArray(w * h) }   // premultiplied r, g, b, weight
        for (i in 0 until w * h) {
            val a = if (mask[i] > 0.5f) 1f else 0f; val c = argb[i]
            cur[0][i] = ((c shr 16) and 255) * a; cur[1][i] = ((c shr 8) and 255) * a; cur[2][i] = (c and 255) * a; cur[3][i] = a
        }
        levels.add(cur); dims.add(intArrayOf(cw, ch))
        while (cw > 1 || ch > 1) {
            val nw = max(1, (cw + 1) / 2); val nh = max(1, (ch + 1) / 2)
            val nxt = Array(4) { FloatArray(nw * nh) }
            for (y in 0 until nh) for (x in 0 until nw) {
                for (dy in 0..1) for (dx in 0..1) {
                    val sx = min(cw - 1, 2 * x + dx); val sy = min(ch - 1, 2 * y + dy); val si = sy * cw + sx
                    for (k in 0..3) nxt[k][y * nw + x] += cur[k][si]
                }
            }
            // normalise to keep weights in 0..1 (colour stays premultiplied)
            for (i in 0 until nw * nh) { val wt = nxt[3][i]; if (wt > 0f) { val s = min(1f, wt) / wt; for (k in 0..3) nxt[k][i] *= s } }
            cur = nxt; cw = nw; ch = nh; levels.add(cur); dims.add(intArrayOf(cw, ch))
        }
        // push: fill missing weight from the coarser level
        for (l in levels.size - 2 downTo 0) {
            val fine = levels[l]; val coarse = levels[l + 1]; val fw = dims[l][0]; val fh = dims[l][1]; val qw = dims[l + 1][0]
            for (y in 0 until fh) for (x in 0 until fw) {
                val i = y * fw + x; val ci = (y / 2) * qw + (x / 2)
                val miss = 1f - min(1f, fine[3][i])
                if (miss <= 0f) continue
                val cwt = coarse[3][ci]
                if (cwt <= 0f) continue
                for (k in 0..2) fine[k][i] += miss * coarse[k][ci] / cwt
                fine[3][i] += miss
            }
        }
        val base = levels[0]
        val out = Array(3) { FloatArray(w * h) }
        for (i in 0 until w * h) { val wt = max(1e-6f, base[3][i]); for (k in 0..2) out[k][i] = base[k][i] / wt }
        // light smoothing outside the mask hides the pull-push blockiness
        val sm = Array(3) { blur(out[it], w, h, max(1.0, min(w, h) / 300.0)) }
        for (i in 0 until w * h) if (mask[i] <= 0.5f) for (k in 0..2) out[k][i] = sm[k][i]
        return out
    }

    // ------------------------------------------------------------------ the try-on
    fun tryOn(
        person: IntArray, pw: Int, ph: Int, pParse: Parse,
        garment: IntArray, gw: Int, gh: Int, gParse: Parse?, kind: GarmentKind,
        shadingStrength: Float = 0.6f,
    ): TryOnResult {
        require(pParse.w == pw && pParse.h == ph) { "parse size" }
        val target = targetRegion(pParse, kind)
        var tn = 0; for (v in target) if (v > 0.5f) tn++
        if (tn < pw * ph * MIN_TARGET_FRACTION)
            throw TryOnException(when (kind) {
                GarmentKind.UPPER -> "Couldn't find a top on the person. Try Full dress, or a photo where the upper body is clearly visible."
                GarmentKind.LOWER -> "Couldn't find trousers / a skirt on the person. Try Full dress, or a full-body photo."
                GarmentKind.FULL -> "Couldn't find the clothes on the person. Use a clear, front-facing full-body photo."
            })
        val (gm, worn) = garmentMask(garment, gw, gh, gParse, kind)
        val tex = extendTexture(garment, gm, gw, gh)

        val tSpans = smoothSpans(rowSpans(target, pw, ph, 0.5f), ph, max(5, ph / 25)) ?: throw TryOnException("Clothes region too small.")
        val gSpans = smoothSpans(rowSpans(gm, gw, gh, 0.5f), gh, max(5, gh / 15)) ?: throw TryOnException("Garment too small in the clothing photo.")
        val tRows = (0 until ph).filter { !tSpans.first[it].isNaN() }; val ty0 = tRows.first(); val ty1 = tRows.last()
        val gRows = (0 until gh).filter { !gSpans.first[it].isNaN() }; val gy0 = gRows.first(); val gy1 = gRows.last()

        // shading from the original clothes: luminance / local mean over the target (normalised blur)
        val lum = FloatArray(pw * ph); val tm = FloatArray(pw * ph)
        for (i in 0 until pw * ph) { val c = person[i]; lum[i] = 0.299f * ((c shr 16) and 255) + 0.587f * ((c shr 8) and 255) + 0.114f * (c and 255); tm[i] = if (target[i] > 0.3f) 1f else 0f }
        val tw = ((ty1 - ty0) + 1).toDouble()
        val sig = max(2.0, tw / 25.0)
        val lm = FloatArray(pw * ph) { lum[it] * tm[it] }
        val bl = blur(lm, pw, ph, sig); val bm = blur(tm, pw, ph, sig)
        // fine scale: removes the old garment's print / pattern so only folds and light remain
        val fs = max(1.5, tw / 140.0)
        val fl = blur(lm, pw, ph, fs); val fm = blur(tm, pw, ph, fs)
        for (i in 0 until pw * ph) lum[i] = if (fm[i] > 1e-3f) fl[i] / fm[i] else lum[i]
        val feather = blur(target, pw, ph, max(0.8, min(pw, ph) / 900.0))

        val out = person.copyOf()
        var covered = 0
        for (y in ty0..ty1) {
            val L = tSpans.first[y]; val R = tSpans.second[y]; if (L.isNaN() || R - L < 1) continue
            val v = if (ty1 > ty0) (y - ty0).toDouble() / (ty1 - ty0) else 0.0
            val gyf = gy0 + v * (gy1 - gy0)
            val gyi = gyf.toInt().coerceIn(gy0, gy1)
            val gL = gSpans.first[gyi]; val gR = gSpans.second[gyi]
            if (gL.isNaN()) continue
            for (x in max(0, L.toInt() - 2)..min(pw - 1, R.toInt() + 2)) {
                val i = y * pw + x
                val a = feather[i]
                if (a < 0.01f) continue
                val u = ((x + 0.5 - L) / (R - L)).coerceIn(0.0, 1.0)
                val gx = gL + u * (gR - gL) - 0.5
                var r = sample(tex[0], gw, gh, gx, gyf); var g = sample(tex[1], gw, gh, gx, gyf); var b = sample(tex[2], gw, gh, gx, gyf)
                val mean = if (bm[i] > 1e-3f) bl[i] / bm[i] else lum[i]
                val ratio = (lum[i] / max(8f, mean)).coerceIn(0.6f, 1.3f)
                val s = 1f + shadingStrength * (ratio - 1f)
                r *= s; g *= s; b *= s
                val c = person[i]
                val nr = (a * r + (1 - a) * ((c shr 16) and 255)).roundToInt().coerceIn(0, 255)
                val ng = (a * g + (1 - a) * ((c shr 8) and 255)).roundToInt().coerceIn(0, 255)
                val nb = (a * b + (1 - a) * (c and 255)).roundToInt().coerceIn(0, 255)
                out[i] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
                if (a > 0.5f) covered++
            }
        }
        val note = if (worn) "Garment taken from the person in the clothing photo." else "Garment cut out from the clothing photo's background."
        return TryOnResult(out, pw, ph, feather, covered.toFloat() / max(1, tn), note, gm, gw, gh)
    }

    private fun sample(ch: FloatArray, w: Int, h: Int, x: Double, y: Double): Float {
        val xx = x.coerceIn(0.0, w - 1.0); val yy = y.coerceIn(0.0, h - 1.0)
        val x0 = xx.toInt(); val y0 = yy.toInt(); val x1 = min(w - 1, x0 + 1); val y1 = min(h - 1, y0 + 1)
        val fx = (xx - x0).toFloat(); val fy = (yy - y0).toFloat()
        val a = ch[y0 * w + x0] * (1 - fx) + ch[y0 * w + x1] * fx
        val b = ch[y1 * w + x0] * (1 - fx) + ch[y1 * w + x1] * fx
        return a * (1 - fy) + b * fy
    }

    @Suppress("unused") private fun absf(v: Float) = abs(v)
}
