from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# 1) SETUP HELPERS (clone repo, download checkpoints, login)

def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n\n{p.stdout}")


def _ensure_repo_cloned(repo_url: str, target_dir: Path) -> None:
    if target_dir.exists() and (target_dir / ".git").exists():
        return
    if target_dir.exists():
        shutil.rmtree(target_dir)
    _run(["git", "clone", repo_url, str(target_dir)])


def _hf_login_if_token(token: Optional[str]) -> None:
    if not token:
        return
    from huggingface_hub import login
    login(token=token)


def _ensure_checkpoints(hf_repo: str, ckpt_dir: Path) -> None:
    """
    Ensures these exist:
      ckpt_dir/model.ckpt
      ckpt_dir/model_config.yaml
      ckpt_dir/assets/mhr_model.pt
    """
    from huggingface_hub import snapshot_download

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    required = [
        ckpt_dir / "model.ckpt",
        ckpt_dir / "model_config.yaml",
        ckpt_dir / "assets" / "mhr_model.pt",
    ]
    if all(p.exists() for p in required):
        return

    shutil.rmtree(ckpt_dir, ignore_errors=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=hf_repo,
        local_dir=str(ckpt_dir),
        local_dir_use_symlinks=False,
    )

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required checkpoint files:\n" + "\n".join(missing))

# 2) LOAD MODELS (SAM-3D + Estimator, YOLO later)

def _load_sam3d(sam_repo_dir: Path, ckpt_dir: Path, device: str):
    import torch
    sys.path.append(str(sam_repo_dir))
    from sam_3d_body.build_models import load_sam_3d_body

    model, model_cfg = load_sam_3d_body(
        str(ckpt_dir / "model.ckpt"),
        device=torch.device(device),
        mhr_path=str(ckpt_dir / "assets" / "mhr_model.pt"),
    )
    return model, model_cfg


def _build_estimator(sam_repo_dir: Path, model, model_cfg):
    # robust import (SAM3DBodyEstimator can be in different modules)
    import importlib
    import inspect
    import pkgutil

    sys.path.append(str(sam_repo_dir))
    import sam_3d_body as sam_pkg

    Estimator = None
    for m in pkgutil.walk_packages(sam_pkg.__path__, prefix=sam_pkg.__name__ + "."):
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue
        if hasattr(mod, "SAM3DBodyEstimator"):
            Estimator = getattr(mod, "SAM3DBodyEstimator")
            break

    if Estimator is None:
        raise RuntimeError("Could not find SAM3DBodyEstimator.")

    sig = inspect.signature(Estimator.__init__)
    kwargs = {}
    for name in sig.parameters:
        if name == "self":
            continue
        if name in ("model", "sam_3d_body_model"):
            kwargs[name] = model
        if name in ("cfg", "model_cfg"):
            kwargs[name] = model_cfg

    return Estimator(**kwargs)

# 3) DETECT PEOPLE (YOLO)

def _detect_people(image_path: Path, weights: str, conf: float, iou: float) -> np.ndarray:
    from ultralytics import YOLO

    yolo = YOLO(weights)
    res = yolo(str(image_path), conf=conf, iou=iou)[0]

    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)

    # COCO class 0 = person
    return xyxy[cls == 0]


def _to_numpy(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# 4) RECONSTRUCT + EXPORT (crop → SAM3D → OBJ)

def _reconstruct_and_export(
    image_path: Path,
    boxes_xyxy: np.ndarray,
    estimator,
    out_dir: Path,
    pad: float,
) -> Tuple[Path, List[Path]]:
    import cv2
    import trimesh

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    H, W = img.shape[:2]

    faces = _to_numpy(estimator.faces).reshape(-1, 3).astype(np.int64)

    indiv_paths: List[Path] = []
    all_verts = []
    all_faces = []
    v_offset = 0

    with tempfile.TemporaryDirectory() as tmpd:
        tmpd = Path(tmpd)

        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            bw, bh = (x2 - x1), (y2 - y1)
            px, py = pad * bw, pad * bh

            xx1 = max(0, int(x1 - px))
            yy1 = max(0, int(y1 - py))
            xx2 = min(W, int(x2 + px))
            yy2 = min(H, int(y2 + py))

            crop = img[yy1:yy2, xx1:xx2].copy()
            crop_path = tmpd / f"crop_{i:02d}.jpg"
            cv2.imwrite(str(crop_path), crop)

            out = estimator.process_one_image(str(crop_path))
            if isinstance(out, list):
                if len(out) == 0:
                    continue
                person = out[0]
            else:
                person = out

            verts = _to_numpy(person["pred_vertices"]).reshape(-1, 3).astype(np.float32)

            # simple scene placement based on bbox center
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            tx = (cx - W / 2) / W * 2.0
            ty = -(cy - H / 2) / H * 2.0
            tz = 0.0
            verts_scene = verts + np.array([tx, ty, tz], dtype=np.float32)

            person_obj = out_dir / f"person_{i:02d}.obj"
            trimesh.Trimesh(vertices=verts_scene, faces=faces, process=False).export(str(person_obj))
            indiv_paths.append(person_obj)

            all_verts.append(verts_scene)
            all_faces.append(faces + v_offset)
            v_offset += verts_scene.shape[0]

    if not all_verts:
        raise RuntimeError("No meshes reconstructed from detected people.")

    merged_obj = out_dir / "all_people.obj"
    merged = trimesh.Trimesh(
        vertices=np.vstack(all_verts),
        faces=np.vstack(all_faces),
        process=False,
    )
    merged.export(str(merged_obj))

    return merged_obj, indiv_paths


# 5) THE MAIN PIPELINE FUNCTION (what run_inference.py calls)


def run_pipeline(
    image: str,
    out_dir: str = "outputs",
    hf_repo: str = "facebook/sam-3d-body-dinov3",
    ckpt_dir: str = "checkpoints/sam-3d-body",
    sam_repo_url: str = "https://github.com/facebookresearch/sam-3d-body.git",
    sam_repo_dir: str = "third_party/sam-3d-body",
    yolo_weights: str = "yolov8x.pt",
    conf: float = 0.20,
    iou: float = 0.50,
    pad: float = 0.15,
    device: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Tuple[Path, List[Path]]:
    """
    End-to-end runner:
      1) download checkpoints
      2) clone SAM-3D repo
      3) load SAM-3D model
      4) detect people with YOLO
      5) reconstruct + export OBJ
    """
    import torch

    image_path = Path(image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # device auto
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # HF login (prefer env var)
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
    _hf_login_if_token(hf_token)

    # checkpoints + repo
    ckpt_dir_p = Path(ckpt_dir)
    _ensure_checkpoints(hf_repo=hf_repo, ckpt_dir=ckpt_dir_p)

    sam_repo_dir_p = Path(sam_repo_dir)
    _ensure_repo_cloned(sam_repo_url, sam_repo_dir_p)

    # model + estimator
    model, model_cfg = _load_sam3d(sam_repo_dir_p, ckpt_dir_p, device=device)
    estimator = _build_estimator(sam_repo_dir_p, model, model_cfg)

    # detect + reconstruct
    boxes = _detect_people(image_path, weights=yolo_weights, conf=conf, iou=iou)
    if boxes.shape[0] == 0:
        raise RuntimeError("No people detected. Try lowering conf/iou or use a clearer image.")

    return _reconstruct_and_export(
        image_path=image_path,
        boxes_xyxy=boxes,
        estimator=estimator,
        out_dir=out_dir_p,
        pad=pad,
    )
