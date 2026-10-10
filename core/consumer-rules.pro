# Consumer rules of :core, applied to both apps. Conservative: shrink Compose/Kotlin/AndroidX,
# but keep native-bridged libraries intact.

# MediaPipe: JNI callbacks from native code, protobuf-lite messages accessed reflectively,
# task options/results built through AutoValue. Keep everything.
-keep class com.google.mediapipe.** { *; }
-keep interface com.google.mediapipe.** { *; }
-keepclassmembers class com.google.mediapipe.** { native <methods>; }

# Protobuf (lite) uses reflection on generated message fields.
-keep class com.google.protobuf.** { *; }
-keepclassmembers class * extends com.google.protobuf.GeneratedMessageLite { *; }

# Flogger (MediaPipe logging) loads its backend reflectively.
-keep class com.google.common.flogger.** { *; }

# ONNX Runtime Java API: the JNI library looks up classes, fields and constructors by name
# (OnnxTensor, OrtSession$Result, OrtException, ...). Keep everything.
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { native <methods>; }

# Keep our pipeline classes readable in stack traces / the on-screen diagnostic line.
-keep class com.vanu.faceswap.core.AiEngine { *; }
-keep class com.vanu.faceswap.core.AiMath { *; }
-keep class com.vanu.faceswap.core.FaceData { *; }
-keep class com.vanu.faceswap.core.FaceDetector { *; }

-keepattributes Signature,*Annotation*,InnerClasses,EnclosingMethod,SourceFile,LineNumberTable
-keepclasseswithmembernames,includedescriptorclasses class * { native <methods>; }

# Compile-time-only annotations / optional deps referenced by the libraries.
-dontwarn javax.annotation.**
-dontwarn com.google.auto.value.**
-dontwarn com.google.errorprone.annotations.**
-dontwarn org.checkerframework.**
-dontwarn com.google.j2objc.annotations.**
-dontwarn com.google.common.**
-dontwarn com.google.mediapipe.**
-dontwarn com.google.protobuf.**
-dontwarn ai.onnxruntime.**
# MediaPipe drags the AutoValue annotation processor (with shaded JavaPoet) onto the runtime
# classpath; it is never used at runtime and references javac-only APIs.
-dontwarn javax.lang.model.**
-dontwarn autovalue.shaded.**
-dontwarn com.google.auto.**

# Keep our exception names readable in the on-screen diagnostic line / logs.
-keepnames class com.vanu.faceswap.core.ImageLoadException
-keepnames class com.vanu.faceswap.core.NoFaceException
-keepnames class com.vanu.faceswap.core.ChecksumException
-keepnames class com.vanu.faceswap.core.StorageException
-keepnames class com.vanu.faceswap.core.ModelMissingException
