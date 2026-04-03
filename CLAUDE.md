# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a visual localization research project (SRTP) that uses the SuperPoint neural network for visual place recognition and camera pose estimation. The system builds a 3D map from images, then localizes query images by solving a Perspective-n-Point (PnP) problem.

## Running the Pipeline

All scripts live in `SuperPointPretrainedNetwork-master/`. Run them from that directory.

**Step 1 — 3D Reconstruction** (one-time, requires a set of scene images):
```bash
cd SuperPointPretrainedNetwork-master
python run_hloc.py
```
Outputs `feats-superpoint-n4096-r1024.h5` and a COLMAP reconstruction folder. Edit the hardcoded `image_dir` and `output_dir` paths inside the script before running.

**Step 2 — Generate 3D point descriptors** (one-time, after reconstruction):
```bash
python step1_generate_descriptors.py
```
Outputs `point3d_to_desc.npy`. Edit `recon_dir` and `features_file` paths inside the script.

**Step 3 — Localize a query image**:
```bash
# Single image
python step2_pnp_localization.py <image_path> --map <reconstruction_dir> --desc point3d_to_desc.npy

# Folder of images
python step2_pnp_localization.py <folder_path> --map <reconstruction_dir> --desc point3d_to_desc.npy

# Live camera
python step2_pnp_localization.py camera --map <reconstruction_dir> --desc point3d_to_desc.npy
```
Outputs `<image>_pose.jpg` (estimated pose axes drawn) and `pose_comparison.jpg` (estimated vs. COLMAP ground truth).

**Run the original SuperPoint demo** (video/webcam point tracking):
```bash
python demo_superpoint.py --weights_path superpoint_v1.pth --input camera
python demo_superpoint.py --weights_path superpoint_v1.pth --input <video_or_image_dir>
```

## Dependencies

No `requirements.txt` exists. Required packages:
- `torch` (PyTorch ≥ 0.4)
- `opencv-python` (≥ 3.4)
- `numpy`, `scipy`, `h5py`, `Pillow`
- `pycolmap` — Python bindings for COLMAP
- `hloc` — [Hierarchical Localization](https://github.com/cvg/Hierarchical-Localization) toolbox

The project uses a conda environment (see `.vscode/settings.json`).

## Architecture

### Core Neural Network (`demo_superpoint.py`)

`SuperPointNet` is a VGG-style encoder with two heads:
- **Detector head**: 65-channel output (64 cells + 1 dustbin), converted to a keypoint heatmap
- **Descriptor head**: 256-dimensional descriptors per spatial location, sampled via bilinear interpolation

`SuperPointFrontend` wraps the model and handles NMS, descriptor normalization, and nearest-neighbor matching. `PointTracker` performs two-way matching for video tracking.

### Localization Pipeline

```
run_hloc.py
  └─ SuperPoint feature extraction → features.h5
  └─ All-pairs matching with LightGlue → matches.h5
  └─ COLMAP SfM reconstruction → reconstruction/ (binary format)

step1_generate_descriptors.py
  └─ For each 3D point: find closest 2D keypoint (≤5px) → assign its SuperPoint descriptor
  └─ Output: point3d_to_desc.npy  (dict: point3D_id → 256-dim descriptor)

step2_pnp_localization.py  (classes: MapManager, PnPSolver, FeatureExtractor, Visualizer, Localizer)
  └─ MapManager: loads reconstruction + point3d_to_desc.npy, extracts camera intrinsics K
  └─ FeatureExtractor: runs SuperPointFrontend, rescales to max 1024px
  └─ MapManager.match_3d_points: cosine similarity matching, threshold 0.7
  └─ PnPSolver: cv2.solvePnPRansac (1000 iters, 10px reprojection error), fallback to EPnP
  └─ Visualizer: draws estimated axes (RGB) and COLMAP ground-truth axes (CMY) on output image
```

### Data Flow and Hardcoded Paths

`step1_generate_descriptors.py` and `run_hloc.py` have **hardcoded Windows paths** pointing to `D:\COLMAP\project1\`. Update these before running on a new machine.

`step2_pnp_localization.py` uses `--map` and `--desc` CLI args (defaults also point to `D:\COLMAP\project1\`).

## Known Issues / Current Status

- PnP accuracy not yet validated — the pipeline runs without errors but pose reliability is uncertain (see latest commit message)
- `sparse_3d_point_cloud.py` is incomplete: COLMAP cannot directly ingest h5 files or 256-dim descriptors
- `online_pnp.py` (chessboard-based approach) was abandoned
- Camera intrinsics are extracted from the COLMAP model assuming a simple pinhole camera (`fx, cx, cy` only); `fy` is assumed equal to `fx`
