// Shared library for both apps: model download (ModelStore), ONNX pipeline (AiCore), MediaPipe
// detection (FaceDetector + face_landmarker.task), image loading/saving (ImageUtils), shared Compose UI.
plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.vanu.faceswap.core"
    compileSdk = 35
    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
    testOptions {
        unitTests.isIncludeAndroidResources = true
        unitTests.all { it.maxHeapSize = "2g" }
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    api(composeBom)
    api("androidx.compose.ui:ui")
    api("androidx.compose.ui:ui-graphics")
    api("androidx.compose.material3:material3")
    api("androidx.compose.material:material-icons-core")
    api("androidx.activity:activity-compose:1.9.3")
    api("androidx.core:core-ktx:1.15.0")
    api("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    api("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.exifinterface:exifinterface:1.3.7")
    api("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // On-device face landmarks (468-point mesh) for detection, 5 keypoints and the face-oval mask
    api("com.google.mediapipe:tasks-vision:0.10.20")
    // ONNX Runtime for the AI models (ArcFace, inswapper, GPEN). 1.22.0: 16 KB-page aligned and
    // ~6.6 MB compressed for arm64 (1.30 is ~12.4 MB, which would push the APK over 24 MB).
    api("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.robolectric:robolectric:4.14.1")
    testImplementation("androidx.test:core:1.6.1")
}
