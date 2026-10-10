package com.vanu.dressup

import android.app.Application
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.vanu.faceswap.core.FaceDetector
import com.vanu.faceswap.core.ImageUtils
import com.vanu.faceswap.core.ModelDownloadController
import com.vanu.faceswap.core.ModelStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class DressUi(
    val person: Bitmap? = null,
    val garment: Bitmap? = null,
    val kind: GarmentKind = GarmentKind.FULL,
    val strict: FilterStrictness = FilterStrictness.RELAXED,
    val busy: String? = null,
    val result: Bitmap? = null,
    val showBefore: Boolean = false,
    val note: String? = null,
    val error: String? = null,
    val message: String? = null,
)

class DressViewModel(app: Application) : AndroidViewModel(app) {
    private val store = ModelStore(File(app.filesDir, "models"), DressModels.ALL)
    val models = ModelDownloadController(store, viewModelScope, DressModels.TOTAL).also { it.check() }
    private val _ui = MutableStateFlow(DressUi())
    val ui: StateFlow<DressUi> = _ui.asStateFlow()

    private var parser: PhotoParser? = null
    private var safety: SafetyFilter? = null

    fun pick(slot: String, uri: Uri) {
        viewModelScope.launch {
            _ui.update { it.copy(busy = "Loading photo…", error = null, message = null) }
            try {
                val bmp = withContext(Dispatchers.IO) { ImageUtils.load(getApplication(), uri, slot, MAX_SIDE) }
                _ui.update { if (slot == "person") it.copy(person = bmp, result = null, busy = null) else it.copy(garment = bmp, result = null, busy = null) }
            } catch (e: Throwable) {
                Log.e(ImageUtils.TAG, "load failed", e)
                _ui.update { it.copy(busy = null, error = "Couldn't open that photo. " + (e.message ?: "")) }
            }
        }
    }

    fun setKind(k: GarmentKind) = _ui.update { it.copy(kind = k, result = null) }
    fun setStrict(s: FilterStrictness) = _ui.update { it.copy(strict = s) }
    fun toggleBefore() = _ui.update { it.copy(showBefore = !it.showBefore) }
    fun clearMessage() = _ui.update { it.copy(message = null) }

    fun tryOn() {
        val s = _ui.value; val person = s.person ?: return; val garment = s.garment ?: return
        viewModelScope.launch {
            _ui.update { it.copy(busy = "Checking photos…", error = null, message = null, result = null, showBefore = false) }
            try {
                val out = withContext(Dispatchers.Default) {
                    val ctx = getApplication<Application>()
                    val sf = safety ?: SafetyFilter(store.file(DressModels.SAFETY)).also { safety = it }
                    fun check(b: Bitmap, what: String) {
                        val sc = sf.score(FaceDetector.toRgb(scaled(b, 512)))
                        Log.i(ImageUtils.TAG, "safety $what = $sc")
                        if (sc > s.strict.threshold) throw SafetyBlockedException(what, sc)
                    }
                    check(person, "your photo"); check(garment, "clothing photo")
                    _ui.update { it.copy(busy = "Finding body and clothes…") }
                    val p = parser ?: PhotoParser(ctx, store.file(DressModels.SEG), store.file(DressModels.POSE)).also { parser = it }
                    val pp = p.parse(person); val gp = p.parse(garment)
                    _ui.update { it.copy(busy = "Fitting the ${s.kind.label.lowercase()}…") }
                    val pw = person.width; val ph = person.height; val gw = garment.width; val gh = garment.height
                    val pPx = IntArray(pw * ph).also { person.getPixels(it, 0, pw, 0, 0, pw, ph) }
                    val gPx = IntArray(gw * gh).also { garment.getPixels(it, 0, gw, 0, 0, gw, gh) }
                    val r = TryOnCore.tryOn(pPx, pw, ph, pp, gPx, gw, gh, gp, s.kind)
                    val bmp = Bitmap.createBitmap(r.argb, pw, ph, Bitmap.Config.ARGB_8888)
                    check(bmp, "result")
                    bmp to r.note
                }
                _ui.update { it.copy(busy = null, result = out.first, note = out.second) }
            } catch (e: SafetyBlockedException) {
                _ui.update { it.copy(busy = null, error = "Blocked by the safety filter (${e.what}, score %.2f). Use regular, fully clothed photos.".format(e.score)) }
            } catch (e: TryOnException) {
                _ui.update { it.copy(busy = null, error = e.message) }
            } catch (e: OutOfMemoryError) {
                _ui.update { it.copy(busy = null, error = "Not enough memory for this photo. Try a smaller photo.") }
            } catch (e: Throwable) {
                Log.e(ImageUtils.TAG, "try-on failed", e)
                _ui.update { it.copy(busy = null, error = "Try-on failed: " + ImageUtils.describe(e)) }
            }
        }
    }

    fun save(sideBySide: Boolean) {
        val s = _ui.value; val r = s.result ?: return
        viewModelScope.launch {
            try {
                val where = withContext(Dispatchers.IO) { saveJpeg(getApplication(), if (sideBySide) beforeAfter(s.person!!, r) else r) }
                _ui.update { it.copy(message = "Saved to $where") }
            } catch (e: Throwable) {
                _ui.update { it.copy(error = "Couldn't save: " + ImageUtils.describe(e)) }
            }
        }
    }

    fun share(ctx: Context) {
        val r = _ui.value.result ?: return
        val dir = File(ctx.cacheDir, "shared").apply { mkdirs() }
        val f = File(dir, "dressup.jpg")
        FileOutputStream(f).use { r.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        val uri = FileProvider.getUriForFile(ctx, ctx.packageName + ".fileprovider", f)
        ctx.startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
            type = "image/jpeg"; putExtra(Intent.EXTRA_STREAM, uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }, "Share"))
    }

    override fun onCleared() { parser?.close(); safety?.close() }

    companion object {
        const val MAX_SIDE = 2048

        fun scaled(b: Bitmap, side: Int): Bitmap {
            val s = side.toDouble() / maxOf(b.width, b.height)
            return if (s >= 1) b else Bitmap.createScaledBitmap(b, maxOf(1, (b.width * s).toInt()), maxOf(1, (b.height * s).toInt()), true)
        }

        fun beforeAfter(a: Bitmap, b: Bitmap): Bitmap {
            val gap = maxOf(8, a.width / 100)
            val out = Bitmap.createBitmap(a.width + b.width + gap, maxOf(a.height, b.height), Bitmap.Config.ARGB_8888)
            Canvas(out).apply { drawColor(android.graphics.Color.WHITE); drawBitmap(a, 0f, 0f, null); drawBitmap(b, (a.width + gap).toFloat(), 0f, null) }
            return out
        }

        fun saveJpeg(context: Context, bmp: Bitmap): String {
            val name = "DressUp_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".jpg"
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, name)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/DressUp")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
                val cr = context.contentResolver
                val uri = cr.insert(MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), values)
                    ?: throw IOException("Couldn't create the image in the gallery.")
                try {
                    cr.openOutputStream(uri)?.use { bmp.compress(Bitmap.CompressFormat.JPEG, 95, it) } ?: throw IOException("Couldn't write the image.")
                    values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0); cr.update(uri, values, null, null)
                } catch (e: Exception) { cr.delete(uri, null, null); throw e }
            } else {
                @Suppress("DEPRECATION")
                val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "DressUp")
                if (!dir.exists() && !dir.mkdirs()) throw IOException("Couldn't create Pictures/DressUp.")
                val f = File(dir, name)
                FileOutputStream(f).use { bmp.compress(Bitmap.CompressFormat.JPEG, 95, it) }
                MediaScannerConnection.scanFile(context, arrayOf(f.absolutePath), arrayOf("image/jpeg"), null)
            }
            return "Pictures/DressUp/$name"
        }
    }
}
