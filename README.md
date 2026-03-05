## Run Options

# Option A — Google Colab 
Use `scripts/colab_runner.py` in Colab:
1. Open Google Colab
2. Paste `scripts/colab_runner.py` into a cell (or upload it)
3. Run → upload an image → download `all_people.obj`

# Option B — Run Locally
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --image path/to/image.jpg --out_dir outputs

End-to-end multi-person 3D human body reconstruction pipeline from a single RGB image.

This model integrates detection and reconstruction models to automatically: 
   1) Detect multiple people in an image
   2) Reconstruct each person as a 3D mesh
   3) Merge all meshes into a scene-aligned 3D model
   4) Export individual .obj files as download

Pipe line

Image → YOLOv8 Person Detection → Bounding Box Cropping → SAM-3D Body Estimation → Mesh Export → Scene Merge

Technical Stack
PyTorch, Ultralytics YOLOv8, Meta SAM-3D-Body, OpenCV, Trimesh, HuggingFace Hub.

Features
Multi-person detection, GPU acceleration (CUDA supported), Automatic bounding box padding, Scene-space mesh alignment, Individual + merged mesh export, Google Colab compatible, Modular structure (production-ready layout)
