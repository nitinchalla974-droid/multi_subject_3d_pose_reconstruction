"""
Colab-only runner.

Purpose:
- Upload an image in Google Colab
- Install dependencies
- Download checkpoints
- Run multi-person 3D reconstruction
- Download the output OBJ(s)

For reusable pipeline code, see: src/.../pipeline.py
For local usage, see: scripts/run_pipeline.py
"""

import torch
print("CUDA available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

from google.colab import files
import os, shutil

os.makedirs("/content/images", exist_ok=True)
uploaded = files.upload()

# move uploads into /content/images
for name in uploaded.keys():
    shutil.move(name, f"/content/images/{name}")

print("Images:", os.listdir("/content/images"))

!pip -q install ultralytics
!pip -q install braceexpand opencv-python-headless pillow tqdm yacs timm einops pytorch-lightning omegaconf hydra-core pyrootutils trimesh roma

# Commented out IPython magic to ensure Python compatibility.
# %cd /content
!rm -rf sam-3d-body
!git clone https://github.com/facebookresearch/sam-3d-body.git

import sys
sys.path.append("/content/sam-3d-body")

import sam_3d_body
print("✅ sam_3d_body import OK")

from huggingface_hub import login
import getpass

token = getpass.getpass("Paste your HF token (input hidden): ")
login(token=token)
print("✅ HF login done")

from huggingface_hub import snapshot_download
import os, shutil

HF_REPO = "facebook/sam-3d-body-dinov3"     # change if needed
CKPT_DIR = "/content/checkpoints/sam-3d-body"

shutil.rmtree(CKPT_DIR, ignore_errors=True)
os.makedirs(CKPT_DIR, exist_ok=True)

snapshot_download(
    repo_id=HF_REPO,
    local_dir=CKPT_DIR,
    local_dir_use_symlinks=False,
)

print("Downloaded files (top):", os.listdir(CKPT_DIR)[:20])

# Required files check
req = {
    "model.ckpt": os.path.join(CKPT_DIR, "model.ckpt"),
    "model_config.yaml": os.path.join(CKPT_DIR, "model_config.yaml"),
    "assets/mhr_model.pt": os.path.join(CKPT_DIR, "assets", "mhr_model.pt"),
}
for k, p in req.items():
    print(k, "exists:", os.path.exists(p), "->", p)

import os, torch
from sam_3d_body.build_models import load_sam_3d_body

CKPT_DIR = "/content/checkpoints/sam-3d-body"
checkpoint_path = os.path.join(CKPT_DIR, "model.ckpt")
mhr_path = os.path.join(CKPT_DIR, "assets", "mhr_model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model, model_cfg = load_sam_3d_body(checkpoint_path, device=device, mhr_path=mhr_path)
print("✅ SAM-3D model loaded")

from ultralytics import YOLO
import os
import numpy as np

img_path = "/content/images/" + sorted(os.listdir("/content/images"))[0]
print("Using image:", img_path)

yolo = YOLO("yolov8x.pt")  # strong detector
res = yolo(img_path, conf=0.20, iou=0.5)[0]

xyxy = res.boxes.xyxy.cpu().numpy()
cls = res.boxes.cls.cpu().numpy().astype(int)
conf = res.boxes.conf.cpu().numpy()

person_xyxy = xyxy[cls == 0]
person_conf = conf[cls == 0]

print("✅ Detected people:", len(person_xyxy))
print(person_xyxy)

import cv2, numpy as np, torch, os, trimesh
from pathlib import Path
import importlib, pkgutil
import sam_3d_body as sam_pkg

# Find SAM3DBodyEstimator class reliably
SAM3DBodyEstimator = None
for m in pkgutil.walk_packages(sam_pkg.__path__, prefix=sam_pkg.__name__ + "."):
    try:
        mod = importlib.import_module(m.name)
    except Exception:
        continue
    if hasattr(mod, "SAM3DBodyEstimator"):
        SAM3DBodyEstimator = getattr(mod, "SAM3DBodyEstimator")
        break
assert SAM3DBodyEstimator is not None, "Could not find SAM3DBodyEstimator."

# Build estimator (handles different ctor signatures)
import inspect
sig = inspect.signature(SAM3DBodyEstimator.__init__)
kwargs = {}
for name in sig.parameters:
    if name == "self":
        continue
    if name in ("model", "sam_3d_body_model"):
        kwargs[name] = model
    if name in ("cfg", "model_cfg"):
        kwargs[name] = model_cfg

estimator = SAM3DBodyEstimator(**kwargs)
print("✅ Estimator created")

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

img_bgr = cv2.imread(img_path)
H, W = img_bgr.shape[:2]

faces = to_numpy(estimator.faces).reshape(-1,3).astype(np.int64)

out_dir = Path("/content/output_mesh_multi")
out_dir.mkdir(parents=True, exist_ok=True)

PAD = 0.15  # bbox padding

meshes = []
v_offset = 0
all_verts = []
all_faces = []

for i, (x1, y1, x2, y2) in enumerate(person_xyxy):
    bw, bh = (x2-x1), (y2-y1)
    px, py = PAD*bw, PAD*bh
    xx1 = max(0, int(x1 - px)); yy1 = max(0, int(y1 - py))
    xx2 = min(W, int(x2 + px)); yy2 = min(H, int(y2 + py))

    crop = img_bgr[yy1:yy2, xx1:xx2].copy()
    crop_path = f"/content/crop_{i:02d}.jpg"
    cv2.imwrite(crop_path, crop)

    out = estimator.process_one_image(crop_path)

    # take first result if list
    if isinstance(out, list):
        if len(out) == 0:
            print("⚠️ No person predicted in crop", i)
            continue
        person = out[0]
    else:
        person = out

    verts = to_numpy(person["pred_vertices"]).reshape(-1,3).astype(np.float32)

    # Simple placement by bbox center (keeps people separated similar to photo layout)
    cx = (x1+x2)/2.0
    cy = (y1+y2)/2.0
    tx = (cx - W/2) / W * 2.0
    ty = -(cy - H/2) / H * 2.0
    tz = 0.0
    verts_scene = verts + np.array([tx, ty, tz], dtype=np.float32)

    # Export individual OBJ
    person_obj = out_dir / f"person_{i:02d}.obj"
    trimesh.Trimesh(vertices=verts_scene, faces=faces, process=False).export(person_obj)

    # Merge into one OBJ (offset faces)
    all_verts.append(verts_scene)
    all_faces.append(faces + v_offset)
    v_offset += verts_scene.shape[0]

print("✅ Individual meshes written to:", str(out_dir))

# merged obj
merged_obj = out_dir / "all_people.obj"
merged = trimesh.Trimesh(vertices=np.vstack(all_verts), faces=np.vstack(all_faces), process=False)
merged.export(merged_obj)

print("✅ Merged OBJ saved:", merged_obj)

from google.colab import files
files.download("/content/output_mesh_multi/all_people.obj")
