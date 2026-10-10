package com.vanu.faceswap.core

import android.os.Build
import android.util.Log
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.CancellationException

// ------------------------------------------------------------------ model download state (shared)

enum class ModelPhase { CHECKING, NEEDED, DOWNLOADING, PAUSED, FAILED, READY }

data class ModelUi(
    val phase: ModelPhase = ModelPhase.CHECKING,
    val done: Long = 0L,
    val total: Long = Models.TOTAL_BYTES,
    val bytesPerSec: Double = 0.0,
    val verifying: String? = null,
    val error: String? = null,
    val errorDetail: String? = null,
)

/** User-facing message for a failed model download. */
fun downloadErrorMessage(e: Throwable): String? = when (e) {
    is StorageException, is ChecksumException -> e.message
    is java.net.UnknownHostException -> "No internet connection. Connect to Wi-Fi and tap Retry."
    is java.net.SocketTimeoutException -> "The connection timed out. Tap Retry — the download resumes where it stopped."
    is java.io.IOException -> "Download interrupted. Tap Retry — it resumes where it stopped."
    else -> "Download failed."
}

/**
 * Drives a [ModelStore] download for a ViewModel: check / start / pause (resumable) / re-download,
 * exposed as a [ModelUi] flow.
 */
class ModelDownloadController(private val store: ModelStore, private val scope: CoroutineScope, private val total: Long) {
    private val _state = MutableStateFlow(ModelUi(total = total))
    val state: StateFlow<ModelUi> = _state.asStateFlow()
    private var job: Job? = null
    @Volatile private var pause = false

    fun check() {
        scope.launch {
            val ready = withContext(Dispatchers.IO) { store.allInstalled() }
            val present = withContext(Dispatchers.IO) { store.bytesPresent() }
            _state.value = ModelUi(if (ready) ModelPhase.READY else if (present > 0) ModelPhase.PAUSED else ModelPhase.NEEDED, done = present, total = total)
        }
    }

    fun start(onReady: () -> Unit = {}) {
        if (job?.isActive == true) return
        pause = false
        _state.update { it.copy(phase = ModelPhase.DOWNLOADING, error = null, errorDetail = null, verifying = null) }
        job = scope.launch {
            try {
                withContext(Dispatchers.IO) {
                    store.downloadAll(cancelled = { pause }) { p ->
                        _state.update { s -> s.copy(done = p.done, total = p.total, bytesPerSec = p.bytesPerSec, verifying = if (p.verifying) p.file else null) }
                    }
                }
                _state.value = ModelUi(ModelPhase.READY, done = total, total = total)
                onReady()
            } catch (e: CancellationException) {
                val present = withContext(Dispatchers.IO) { store.bytesPresent() }
                _state.update { it.copy(phase = ModelPhase.PAUSED, done = present, verifying = null, bytesPerSec = 0.0) }
            } catch (e: Throwable) {
                Log.e(ImageUtils.TAG, "model download failed", e)
                val present = withContext(Dispatchers.IO) { store.bytesPresent() }
                _state.update { it.copy(phase = ModelPhase.FAILED, done = present, verifying = null, bytesPerSec = 0.0,
                    error = downloadErrorMessage(e), errorDetail = ImageUtils.describe(e)) }
            }
        }
    }

    fun pause() { pause = true }

    fun redownload() {
        scope.launch {
            withContext(Dispatchers.IO) { store.clear() }
            _state.value = ModelUi(ModelPhase.NEEDED, total = total)
            start()
        }
    }
}

// ------------------------------------------------------------------ shared composables

@Composable
fun AppTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val ctx = LocalContext.current
    val scheme = when {
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> if (dark) dynamicDarkColorScheme(ctx) else dynamicLightColorScheme(ctx)
        dark -> darkColorScheme()
        else -> lightColorScheme()
    }
    MaterialTheme(colorScheme = scheme, content = content)
}

@Composable
fun PhotoSlot(
    title: String, subtitle: String, image: android.graphics.Bitmap?, enabled: Boolean,
    modifier: Modifier = Modifier, onGallery: () -> Unit, onCamera: (() -> Unit)? = null
) {
    Card(modifier = modifier, border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)) {
        Column(Modifier.padding(8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(title, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            Text(subtitle, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(6.dp))
            Box(
                Modifier
                    .fillMaxWidth()
                    .aspectRatio(0.8f)
                    .clip(RoundedCornerShape(10.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant)
                    .clickable(enabled = enabled, onClick = onGallery),
                contentAlignment = Alignment.Center
            ) {
                if (image != null) {
                    val bmp = remember(image) { image.asImageBitmap() }
                    Image(bmp, contentDescription = title, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
                } else {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Icon(Icons.Filled.Add, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text("Tap to pick", style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                TextButton(onClick = onGallery, enabled = enabled) { Text("Gallery") }
                if (onCamera != null) TextButton(onClick = onCamera, enabled = enabled) { Text("Camera") }
            }
        }
    }
}

fun mb(b: Long) = "%.0f MB".format(b / 1e6)

private fun countWord(n: Int) = when (n) { 1 -> "one model"; 2 -> "two models"; 3 -> "three models"; else -> "$n models" }

/** First-run model download card. [extraNote] is shown under the file list (e.g. optional extras). */
@Composable
fun ModelSetup(m: ModelUi, specs: List<ModelSpec>, onDownload: () -> Unit, onPause: () -> Unit, extraNote: String? = null) {
    val total = specs.sumOf { it.bytes }
    Card(modifier = Modifier.fillMaxWidth(), border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("One-time setup", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            Text("The AI needs ${countWord(specs.size)} (${mb(total)} in total). They are downloaded once and kept " +
                "privately inside the app. After that the app works completely offline and your photos never leave your phone.",
                style = MaterialTheme.typography.bodyMedium)
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                for (s in specs) Row(Modifier.fillMaxWidth()) {
                    Text("• " + s.label, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                    Text(mb(s.bytes), style = MaterialTheme.typography.bodySmall)
                }
            }
            extraNote?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.Info, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(18.dp))
                Spacer(Modifier.size(6.dp))
                Text("Wi-Fi recommended — this is a large download. Needs about ${mb(total + 50_000_000L)} of free storage. " +
                    "If the connection drops you can resume where it stopped.", style = MaterialTheme.typography.bodySmall)
            }
            val frac = if (m.total > 0) (m.done.toFloat() / m.total).coerceIn(0f, 1f) else 0f
            if (m.phase == ModelPhase.DOWNLOADING || m.done > 0) {
                LinearProgressIndicator(progress = { frac }, modifier = Modifier.fillMaxWidth())
                val line = when {
                    m.verifying != null -> "Verifying ${m.verifying} (SHA-256)…"
                    m.phase == ModelPhase.DOWNLOADING -> buildString {
                        append("${mb(m.done)} of ${mb(m.total)}")
                        if (m.bytesPerSec > 1000) {
                            append(" · %.1f MB/s".format(m.bytesPerSec / 1e6))
                            val left = ((m.total - m.done) / m.bytesPerSec).toLong()
                            append(" · about " + if (left >= 60) "${left / 60} min left" else "$left s left")
                        }
                    }
                    else -> "${mb(m.done)} of ${mb(m.total)} downloaded"
                }
                Text(line, style = MaterialTheme.typography.bodySmall)
            }
            m.error?.let { err ->
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer,
                    contentColor = MaterialTheme.colorScheme.onErrorContainer), modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(10.dp)) {
                        Text(err, style = MaterialTheme.typography.bodyMedium)
                        m.errorDetail?.let { Text(it, style = MaterialTheme.typography.labelSmall) }
                    }
                }
            }
            when (m.phase) {
                ModelPhase.CHECKING -> CircularProgressIndicator(Modifier.align(Alignment.CenterHorizontally))
                ModelPhase.DOWNLOADING -> OutlinedButton(onClick = onPause, modifier = Modifier.fillMaxWidth()) { Text("Pause") }
                ModelPhase.NEEDED -> Button(onClick = onDownload, modifier = Modifier.fillMaxWidth().height(52.dp)) {
                    Text("Download ${mb(total)}", style = MaterialTheme.typography.titleMedium) }
                ModelPhase.PAUSED -> Button(onClick = onDownload, modifier = Modifier.fillMaxWidth().height(52.dp)) {
                    Text("Resume download", style = MaterialTheme.typography.titleMedium) }
                ModelPhase.FAILED -> Button(onClick = onDownload, modifier = Modifier.fillMaxWidth().height(52.dp)) {
                    Text("Retry", style = MaterialTheme.typography.titleMedium) }
                ModelPhase.READY -> {}
            }
            Text("Every file is checked against its SHA-256 checksum before use. Models: ArcFace and inswapper (InsightFace) and " +
                "GPEN (Alibaba DAMO), from the FaceFusion model repository — licensed for personal, non-commercial use; " +
                "safety filter: Falconsai NSFW classifier (Apache-2.0); body pose and hair/skin/clothes parts: Google MediaPipe (Apache-2.0).",
                style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

/** Error card with the small-print exception detail line. */
@Composable
fun ErrorCard(msg: String, detail: String?, action: (@Composable () -> Unit)? = null) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer,
        contentColor = MaterialTheme.colorScheme.onErrorContainer), modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Filled.Warning, contentDescription = null); Spacer(Modifier.size(10.dp))
            Column {
                Text(msg, style = MaterialTheme.typography.bodyMedium)
                detail?.let { Spacer(Modifier.height(4.dp)); Text(it, style = MaterialTheme.typography.labelSmall) }
                action?.invoke()
            }
        }
    }
}
