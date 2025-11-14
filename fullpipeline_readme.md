
# Full Pipeline: From Synced Videos to 3D Body & Hand Poses

This guide is for users who want to run the **full pipeline** and obtain **3D body and hand poses** from synced multi-view videos.

---

## 1. Requirements

### 1.1 Camera Intrinsics

Place your intrinsics under:

```text
camera_poses/<experiment_name>/camera_intrinsics/
````

* One JSON file **per camera**, e.g.:

  ```text
  camera_poses/visceral/camera_intrinsics/gopro1_synced_intrinsics.json
  ```

* These files can be generated using the
  👉 [calibration](https://github.com/BAL-ROCS-BUT-COOL/calibration) repository.

---

### 1.2 Camera Extrinsics

Place your extrinsics under:

```text
camera_poses/<experiment_name>/camera_extrinsics/aligned_poses.json
```

Example:

```text
camera_poses/visceral/camera_extrinsics/aligned_poses.json
```

How to generate:

1. First create `camera_poses.json` using the
   👉 [calibration](https://github.com/BAL-ROCS-BUT-COOL/calibration) repo.
2. Then create `aligned_poses.json` using the
   👉 [orx 3D reconstruction](https://github.com/BAL-ROCS-BUT-COOL/orx-3d-reconstruction) repo.

---

### 1.3 Input Videos

Place your synced clips under:

```text
inputs/<experiment_name>/
    gopro1_synced_clips.mp4
    gopro2_synced_clips.mp4
    ...
```

* Use `extract_synced_videos.py` from the
  👉 [RocSync](https://github.com/BAL-ROCS-BUT-COOL/RocSync/tree/main/sw/evaluation) repo
  to generate these synced clips.
* The synced clips **must** follow the naming convention:

  ```text
  goproX_synced_clips.mp4
  ```

  If you use a different naming scheme, the pipeline may not work out of the box.

---

## 2. Usage

1. Open `full_pipeline.py`.
2. Set the **near-field** and **far-field** cameras as needed
   (see the configuration section near line 55).
3. Run the pipeline:

   ```bash
   python full_pipeline.py --data-input "<experiment_name>"
   ```

---

## 3. Outputs

After running the pipeline, you will find the results under:

```text
output_3d/<experiment_name>/
```

Specifically:

* `body_poses_3d.npz` – 3D **body** poses
* `hand_poses_3d.npz` – 3D **hand** poses


---

## 4. Getting SMPL-X Meshes

To convert the 3D body and hand poses (NPZ files) to **SMPL-X meshes**, follow:

👉 [Stickman-to-SMPLX](https://github.com/BAL-ROCS-BUT-COOL/Stickman-to-SMPLX)


