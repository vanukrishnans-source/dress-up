package com.vanu.dressup

import com.vanu.faceswap.core.ModelSpec

/** First-run download (114 MB): this app's own GitHub release first, Collage Video's copy and Google's CDN as mirrors. */
object DressModels {
    const val OWN = "https://github.com/vanukrishnans-source/dress-up/releases/download/models-v1/"
    private const val CV = "https://github.com/vanukrishnans-source/collage-video/releases/download/models-v1/"

    val SAFETY = ModelSpec("nsfw_vit_int8.onnx", "Safety filter (NSFW classifier)", 88_639_960L,
        "b6734b6fdc6be4edac08ffd91b07c7b2bc800c6a93c79b029c865bde2fb43627", listOf(OWN + "nsfw_vit_int8.onnx", CV + "nsfw_vit_int8.onnx"))
    val POSE = ModelSpec("pose_landmarker_full.task", "Body pose (MediaPipe)", 9_398_198L,
        "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad", listOf(OWN + "pose_landmarker_full.task", CV + "pose_landmarker_full.task",
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"))
    val SEG = ModelSpec("selfie_multiclass_256x256.tflite", "Clothes / skin / hair parts (MediaPipe)", 16_371_837L,
        "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0", listOf(OWN + "selfie_multiclass_256x256.tflite", CV + "selfie_multiclass_256x256.tflite",
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"))

    val ALL = listOf(SAFETY, POSE, SEG)
    val TOTAL = ALL.sumOf { it.bytes }
}
