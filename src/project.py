from pathlib import Path
from pose import PoseData
from ui import ProjectWindow
from camera import Cameras
from collections import defaultdict


class Project:
    """
    This class is responsible for the main application logic, since that is working on a project.
    It provides functions for performing all actions you can perform on the project and makes sure all other components are updated accordingly.
    """
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg'}
    VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}
    
    def __init__(self, app, dataset_name, *, initial_camera_name=None, initial_frame_idx=None):
        """
        Initialize the application for the given dataset.
        """
        self.app = app
        self.dark_mode = False
        self.dataset_name = dataset_name
        self.window = None
        self.video_path = Path("inputs") / self.dataset_name
        # Try to load manifest (preferred)
        self.manifest = self._load_manifest()
        self.media_map = {}  # for video: {camera_stem: Path}
        self._manifest_foto_files = []  # for fotos: [Path, ...]

        if self.manifest:
            mode = self.manifest.get("mode")
            media = self.manifest.get("media", [])
            if mode == "video":
                # do NOT rely on inputs/ — use media_map
                self.foto_mode = False
                self.foto_index = {}
                self.media_map = {Path(p).stem: Path(p) for p in media}
                cameras_list = sorted(self.media_map.keys())
            elif mode == "foto":
                # foto mode: we’ll build foto_index from the provided file list
                self.foto_mode = True
                self._manifest_foto_files = [Path(p) for p in media]
                self.foto_index = self._build_foto_camera_index()  # uses _iter_files()
                cameras_list = sorted(self.foto_index.keys())
            else:
                # Unknown manifest -> fall back to old behavior
                self.foto_mode = self._is_foto_mode()
                self.foto_index = self._build_foto_camera_index() if self.foto_mode else {}
                cameras_list = sorted(self.foto_index.keys()) if self.foto_mode else self.available_cameras()
        else:
            # No manifest -> legacy behavior (scan inputs/<dataset>)
            self.foto_mode = self._is_foto_mode()
            self.foto_index = self._build_foto_camera_index() if self.foto_mode else {}
            cameras_list = sorted(self.foto_index.keys()) if self.foto_mode else self.available_cameras()



        self.cameras = Cameras(
            self.dataset_name,
            cameras=cameras_list,
            foto_mode=self.foto_mode,
            foto_index=self.foto_index,
            media_map=self.media_map,   # <-- NEW
        )
        if not self.cameras.data:
        # No cameras found — fail fast with a helpful message
            raise RuntimeError(
                f"No cameras found for project '{self.dataset_name}'. "
                "If you're using the new manifest flow, ensure output_3d/<project>/project.json "
                "exists and its 'media' entries point to real files. "
                "If you're using legacy inputs/, ensure inputs/<project>/ contains videos/images."
            )
        
    
        self.dataset = PoseData(self)

        print(self.available_cameras())
        self.current_camera = self.cameras.get(0)
        self._sticky_frame_idx = getattr(self.current_camera, "current_frame_idx", 0)
        self.current_hand = 0
        self.current_keypoint = 0
        self.keypoint_advance = 0
        self.current_person = 0
        self.frame_step = 1 if self.foto_mode else 30
        self.keypoints_hidden = False

        if initial_camera_name is not None:
            self._select_camera_by_name(initial_camera_name)

        self.window = ProjectWindow(self)
        
        # If a starting frame is requested, jump there and render
        if initial_frame_idx is not None and hasattr(self.current_camera, "goto_frame"):
            # Clamp safely into range if needed
            try:
                # Safely cap within [0, frame_count-1], but keep exact index (no frame_step rounding)
                max_idx = max(0, self.current_camera.frame_count - 1)
                target = min(max(0, int(initial_frame_idx)), max_idx)
                self.current_camera.goto_frame(target)
                self._sticky_frame_idx = getattr(self.current_camera, "current_frame_idx", target)
            except Exception:
                # If camera is not ready or fails, fall back to 0
                self.current_camera.goto_frame(0)
            # Render the requested frame
            if hasattr(self.window, "canvas") and hasattr(self.window.canvas, "viewport"):
                self.window.canvas.viewport.render_current_frame()


    def _manifest_path(self) -> Path:
        return Path("output_3d") / self.dataset_name / "project.json"

    def _load_manifest(self):
        """
        Return dict with keys:
        - mode: "video" | "foto"
        - media: [absolute file paths]
        or None if no manifest.
        """
        mp = self._manifest_path()
        if not mp.exists():
            return None

        try:
            import json
            raw = json.loads(mp.read_text())
        except Exception:
            return None

        media = raw.get("media", [])
        base = mp.parent

        resolved_media = []
        for p in media:
            p_path = Path(p)
            if not p_path.is_absolute():
                p_path = (base / p_path).resolve()
            resolved_media.append(str(p_path))

        raw["media"] = resolved_media
        return raw
    
    # ---------- new helper ----------
    def _select_camera_by_name(self, camera_name: str):
        names = self.available_cameras()
        if camera_name in names:
            prev_idx = getattr(self.current_camera, "current_frame_idx", getattr(self, "_sticky_frame_idx", 0))
            idx = names.index(camera_name)
            self.current_camera = self.cameras.get(idx)

            if hasattr(self.current_camera, "goto_frame") and hasattr(self.current_camera, "frame_count"):
                max_idx = max(0, self.current_camera.frame_count - 1)
                target = max(0, min(prev_idx, max_idx))
                if self.frame_step > 1:
                    target -= (target % self.frame_step)
                self.current_camera.goto_frame(target)
                self._sticky_frame_idx = target

            # Render then reset
            if getattr(self, "window", None) and hasattr(self.window, "canvas") and hasattr(self.window.canvas, "viewport"):
                self.window.canvas.viewport.render_camera_change()
                self._reset_view_on_camera_change()
            
    def available_cameras(self):
        """
        Return a list of available cameras for the current project.

        - In foto mode: unique prefixes before the first underscore from image filenames.
        - In video mode: base names (stem) of video files in the folder.
        """
        if self.foto_mode:
            # Keys are camera names like 'gopro10', 'gopro11', ...
            return sorted(self.foto_index.keys())

        # video mode: include only recognized video files
        cameras = []
        for camera_file in self._iter_files():
            if camera_file.suffix.lower() in self.VIDEO_EXTS:
                cameras.append(camera_file.stem)
        return sorted(cameras)

    def change_camera(self, index):
        prev_idx = getattr(self.current_camera, "current_frame_idx", getattr(self, "_sticky_frame_idx", 0))
        self.current_camera = self.cameras.get(index)

        if hasattr(self.current_camera, "goto_frame") and hasattr(self.current_camera, "frame_count"):
            max_idx = max(0, self.current_camera.frame_count - 1)
            target = max(0, min(prev_idx, max_idx))
            if self.frame_step > 1:
                target -= (target % self.frame_step)
            self.current_camera.goto_frame(target)
            self._sticky_frame_idx = target

        # Draw the new camera view first
        self.window.canvas.viewport.render_camera_change()
        # Then reset zoom/pan (only on camera change)
        self._reset_view_on_camera_change()

    def change_person(self, index):
        """
        Change the current person to the one at the given index.
        """
        self.current_person = index
        self.window.canvas.viewport.draw()

    def change_hand(self, index):
        """
        Change the current hand to the one at the given index.
        """
        self.current_hand = index
        self.window.canvas.viewport.draw()

    def save(self):
        """
        Save the current dataset to disk.
        """
        self.dataset.save()

    def goto_last_frame(self):
        """
        Goes to the last frame of the current camera.
        """
        self.current_camera.goto_frame(self.current_camera.frame_count - 1 - (self.current_camera.frame_count - 1) % self.frame_step)
        self._sticky_frame_idx = getattr(self.current_camera, "current_frame_idx", self._sticky_frame_idx)
        self.window.canvas.viewport.render_current_frame()

    def next_frame(self):
        """
        Goes to the next frame of the current camera.
        """
        self.current_camera.goto_frame(self.current_camera.current_frame_idx + self.frame_step)
        self._sticky_frame_idx = getattr(self.current_camera, "current_frame_idx", self._sticky_frame_idx)
        self.window.canvas.viewport.render_current_frame()

    def prev_frame(self):
        """
        Goes to the previous frame of the current camera.
        """
        self.current_camera.goto_frame(self.current_camera.current_frame_idx - self.frame_step)
        self._sticky_frame_idx = getattr(self.current_camera, "current_frame_idx", self._sticky_frame_idx)
        self.window.canvas.viewport.render_current_frame()

    def goto_first_frame(self):
        """
        Goes to the first frame of the current camera.
        """
        self.current_camera.goto_frame(0)
        self._sticky_frame_idx = getattr(self.current_camera, "current_frame_idx", self._sticky_frame_idx)
        self.window.canvas.viewport.render_current_frame()

    def next_keypoint(self):
        """
        Advances to the next keypoint.
        """
        self.set_current_keypoint(self.current_keypoint + 1)

    def prev_keypoint(self):
        """
        Goes back to the previous keypoint.
        """
        self.set_current_keypoint(self.current_keypoint - 1)

    def resize_keypoints(self, size: int):
        """
        Resizes the keypoints.
        """
        self.window.canvas.viewport.set_radius(float(size)/10.0)
        self.window.canvas.viewport.draw()

    def set_current_keypoint(self, idx):
        """
        Sets the current keypoint index.
        """
        if 0 <= idx <= 20:
            self.current_keypoint = idx
        elif idx < 0:
            self.current_keypoint = 20
        elif idx > 20:
            self.current_keypoint = 0
        self.window.canvas.keypoint_picker.draw()
        self.window.canvas.viewport.draw()

    def place_keypoint(self, x, y):
        """
        Places a keypoint at the given coordinates.
        """
        self.dataset.get_pose(self.current_person, self.current_camera, self.current_hand).place_keypoint(self.current_keypoint, x, y)
        if self.keypoint_advance != 0:
            self.set_current_keypoint(self.current_keypoint + self.keypoint_advance)
        self.window.canvas.viewport.draw()

    def delete_keypoint(self):
        """
        Deletes the current keypoint.
        """
        self.dataset.get_pose(self.current_person, self.current_camera, self.current_hand).remove_keypoint(self.current_keypoint)
        if self.keypoint_advance != 0:
            self.set_current_keypoint(self.current_keypoint + self.keypoint_advance)
        self.window.canvas.viewport.draw()

    def toggle_keypoint_visibility(self):
        """
        Toggles the visibility of the keypoints.
        """
        self.keypoints_hidden = not self.keypoints_hidden
        self.window.toolbar.visibility_button.setIcon(self.window.toolbar.visibility_icons[self.keypoints_hidden])
        self.window.canvas.viewport.draw()

    def flip_hand_side(self):
        """
        Flips the data of the two hands.
        """
        self.dataset.flip_hands(self.current_person, self.current_camera)
        self.window.canvas.viewport.draw()

    def swap_people(self, other):
        self.dataset.flip_person(self.current_person, other, camera=self.current_camera)
        self.window.canvas.viewport.draw()

    # ---------- helpers ----------

    def _iter_files(self):
        """Yield files for the dataset:
        - If manifest present (video): yield media_map values
        - If manifest present (foto): yield the foto file list
        - Else: scan inputs/<dataset>
        """
        if self.manifest:
            if not self.foto_mode:
                return list(self.media_map.values())
            else:
                return list(self._manifest_foto_files)

        # legacy fallback (no manifest)
        if not self.video_path.exists():
            return []
        return [p for p in self.video_path.iterdir() if p.is_file()]

    def _is_foto_mode(self) -> bool:
        """
        Foto mode if the folder exists, has at least one file,
        and **every** file is an image (png/jpg/jpeg).
        """
        files = self._iter_files()
        if not files:
            return False
        exts = {p.suffix.lower() for p in files}
        # Only allow known image extensions
        return all(ext in self.IMAGE_EXTS for ext in exts)

    def _build_foto_camera_index(self) -> dict:
        """
        Build a mapping {camera_name: [image files]} where camera_name is the
        part BEFORE the first underscore in the filename.
        Example: gopro10_343.jpg -> camera 'gopro10'
        """
        index = defaultdict(list)
        for f in self._iter_files():
            if f.suffix.lower() not in self.IMAGE_EXTS:
                continue
            stem = f.stem
            # Split on the first underscore; if none, use entire stem as camera name
            camera = stem.split('_', 1)[0] if '_' in stem else stem
            index[camera].append(f)
        # Sort file lists for reproducibility (optional)
        for cam in index:
            index[cam].sort()
        return dict(index)
    
    def _reset_view_on_camera_change(self):
        """
        Reset zoom/pan to full frame and clear active pan/zoom tools.
        Call ONLY when changing cameras.
        """
        canvas = getattr(self.window, "canvas", None)
        if not canvas:
            return

        # 1) Make sure pan/zoom tools are deactivated
        nav = getattr(canvas, "nav_toolbar", None)
        if nav:
            # Best-effort: turn off pan/zoom if they were active
            try: nav.pan(False)
            except Exception: pass
            try: nav.zoom(False)
            except Exception: pass

        # 2) Prefer the app's canonical reset
        if hasattr(canvas, "reset_view"):
            try:
                canvas.reset_view()
                return
            except Exception:
                pass

        # 3) Fallback: reset axes to the image extent
        vp = getattr(canvas, "viewport", None)
        ax = getattr(vp, "axes", None) if vp else None
        if ax is not None:
            try:
                # If an image is drawn, use its extent
                ims = ax.get_images()
                if ims:
                    x0, x1, y0, y1 = ims[-1].get_extent()
                    ax.set_xlim(x0, x1)
                    ax.set_ylim(y0, y1)
                else:
                    # Generic autoscale fallback
                    ax.relim()
                    ax.autoscale_view()
            except Exception:
                pass
            # Ensure canvas updates
            if hasattr(canvas, "draw"):
                canvas.draw()
