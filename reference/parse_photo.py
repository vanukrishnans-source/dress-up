"""Desktop stand-in for the app's MediaPipe step (same .tflite/.task models): parts + pose for one photo.
Writes <out>/rgb.png (resized to <= 2048 px), maps.f32 (5 x h x w: hair, skin, face, clothes, background) and meta.json."""
import sys, os, json, numpy as np
from PIL import Image, ImageOps
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

M = os.environ.get("MODELS", "/workspace/collage-video-app/models_build")
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
s = min(1.0, 2048 / max(im.size))
if s < 1: im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
im.save(f"{out}/rgb.png")
np.asarray(im).tofile(f"{out}/rgb.raw")
W, H = im.size
arr = np.ascontiguousarray(np.asarray(im))
mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
# letterbox to a square (grey) so the 256x256 parts model doesn't see a squashed person; same as the app
S = max(W, H); ox, oy = (S - W) // 2, (S - H) // 2
sq = Image.new("RGB", (S, S), (127, 127, 127)); sq.paste(im, (ox, oy))
sqimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(sq)))
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=f"{M}/selfie_multiclass_256x256.tflite"),
    running_mode=vision.RunningMode.IMAGE, output_confidence_masks=True, output_category_mask=False))
r = seg.segment(sqimg)
def crop(m):
    m = np.squeeze(m.numpy_view()).astype(np.float32)
    if m.shape != (S, S): m = np.asarray(Image.fromarray(m, mode="F").resize((S, S), Image.BILINEAR))
    return m[oy:oy + H, ox:ox + W]
conf = [crop(m) for m in r.confidence_masks]
bg, hair, body, face, clothes, others = conf[:6]
np.stack([hair, body, face, clothes, bg, others]).astype("<f4").tofile(f"{out}/maps.f32")
pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=f"{M}/pose_landmarker_full.task"),
    running_mode=vision.RunningMode.IMAGE, num_poses=1, min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4))
pr = pose.detect(mpimg)
meta = {"w": W, "h": H, "shoulderY": None, "hipY": None, "faceVisible": False}
if pr.pose_landmarks:
    l = pr.pose_landmarks[0]
    v = lambda i: (l[i].visibility or 0)
    if v(11) > 0.3 and v(12) > 0.3: meta["shoulderY"] = (l[11].y + l[12].y) / 2 * H
    if v(23) > 0.3 and v(24) > 0.3: meta["hipY"] = (l[23].y + l[24].y) / 2 * H
    meta["faceVisible"] = v(0) > 0.5 and (v(2) > 0.5 or v(5) > 0.5)
json.dump(meta, open(f"{out}/meta.json", "w"))
print(src, meta)
