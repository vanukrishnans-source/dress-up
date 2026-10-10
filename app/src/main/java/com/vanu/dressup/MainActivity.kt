package com.vanu.dressup

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.vanu.faceswap.core.AppTheme
import com.vanu.faceswap.core.ErrorCard
import com.vanu.faceswap.core.ModelPhase
import com.vanu.faceswap.core.ModelSetup
import com.vanu.faceswap.core.PhotoSlot

/** "Dress Up": your photo on the left, a dress on the right, Try On. Only the clothes change. */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent { AppTheme { DressApp() } }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DressApp(vm: DressViewModel = viewModel()) {
    val models by vm.models.state.collectAsStateWithLifecycle()
    val view = LocalView.current
    val keepOn = models.phase == ModelPhase.DOWNLOADING
    DisposableEffect(keepOn) { view.keepScreenOn = keepOn; onDispose { view.keepScreenOn = false } }
    Scaffold(topBar = { TopAppBar(title = { Text("Dress Up", fontWeight = FontWeight.SemiBold) }) }) { padding ->
        Column(
            Modifier.fillMaxSize().padding(padding).verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if (models.phase != ModelPhase.READY) {
                ModelSetup(models, DressModels.ALL, onDownload = { vm.models.start() }, onPause = { vm.models.pause() })
            } else {
                DressContent(vm)
            }
            Text("Everything runs on your phone. Photos never leave the device; the internet is only used once to download the models.",
                style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp))
        }
    }
}

@Composable
fun DressContent(vm: DressViewModel) {
    val s by vm.ui.collectAsStateWithLifecycle()
    val ctx = LocalContext.current
    val idle = s.busy == null
    val pickPerson = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { it?.let { u -> vm.pick("person", u) } }
    val pickGarment = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { it?.let { u -> vm.pick("garment", u) } }
    val req = PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)

    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        PhotoSlot("Your photo", "full body, facing front", s.person, idle, Modifier.weight(1f), onGallery = { pickPerson.launch(req) })
        PhotoSlot("Dress", "product, flat lay or worn", s.garment, idle, Modifier.weight(1f), onGallery = { pickGarment.launch(req) })
    }
    Text("What to change", style = MaterialTheme.typography.titleSmall)
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        for (k in listOf(GarmentKind.FULL, GarmentKind.UPPER, GarmentKind.LOWER))
            FilterChip(selected = s.kind == k, onClick = { vm.setKind(k) }, enabled = idle, label = { Text(k.label) })
    }
    Button(onClick = { vm.tryOn() }, enabled = idle && s.person != null && s.garment != null,
        modifier = Modifier.fillMaxWidth().height(56.dp)) { Text("Try On", style = MaterialTheme.typography.titleMedium) }
    if (s.busy != null) Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        CircularProgressIndicator(Modifier.size(22.dp)); Text(s.busy!!)
    }
    s.error?.let { ErrorCard(it, null) }
    s.message?.let { Text(it, color = MaterialTheme.colorScheme.primary) }

    val result = s.result
    if (result != null && s.person != null) {
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(if (s.showBefore) "Before (tap to see after)" else "After (tap to see before)", style = MaterialTheme.typography.titleSmall)
                val shown = if (s.showBefore) s.person!! else result
                val img = remember(shown) { shown.asImageBitmap() }
                Image(img, contentDescription = "result", contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxWidth().height(520.dp).clickable { vm.toggleBefore() })
                s.note?.let { Text(it, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = { vm.save(false) }, modifier = Modifier.weight(1f)) { Text("Save") }
                    OutlinedButton(onClick = { vm.share(ctx) }, modifier = Modifier.weight(1f)) { Text("Share") }
                }
                OutlinedButton(onClick = { vm.save(true) }, modifier = Modifier.fillMaxWidth()) { Text("Save before | after") }
                Text("Saved at ${result.width}×${result.height} to Pictures/DressUp.", style = MaterialTheme.typography.labelSmall)
            }
        }
    }
    Text("Safety filter (always on)", style = MaterialTheme.typography.titleSmall)
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        for (f in FilterStrictness.entries) FilterChip(selected = s.strict == f, onClick = { vm.setStrict(f) }, enabled = idle, label = { Text(f.label) })
    }
    Text(s.strict.help + " Clothes are only ever replaced by the garment you pick.", style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant)
}
