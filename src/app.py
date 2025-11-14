import argparse
from project import Project
from PyQt5.QtWidgets import QApplication, QErrorMessage
from ui.launcher import Launcher
import sys
import os
from pathlib import Path
import shutil
import json
import cv2
import signal
import traceback

class App(QApplication):
    """
    This class represents the main application for pose correction.
    It manages the creation and opening of projects, as well as startup of the application.
    """
    def __init__(self):
        """
        Initializes the application.

        Launches the launcher if no arguments are provided.
        Otherwise, opens the project specified by the first argument.
        """
        super().__init__(sys.argv)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("[App] SIGINT (Ctrl+C) enabled")
        dataset_name = self.parse_args().dataset
        if dataset_name is not None:
            self.open_project(dataset_name)
        else:
            self.launcher = Launcher(self)
            print("Launcher created")
            self.launcher.show()
            print("Launcher shown")

    def open_project(self, dataset_name, initial_camera_name=None, initial_frame_idx=None):
        print("Dataset name:", dataset_name)
        try:
            self.current_project = Project(
                self,
                dataset_name,
                initial_camera_name=initial_camera_name,
                initial_frame_idx=initial_frame_idx,
            )
            win = getattr(self.current_project, "window", None)
            if win is not None:
                win.show()
                print("Project window shown")
            else:
                print("[open_project] Warning: project.window is None")
            return True

        except Exception as e:
            print(f"[open_project] Failed to open project '{dataset_name}': {e}")
            traceback.print_exc()  # <<< show where the error comes from
            self.current_project = None
            return False


    def get_project_names(self, root="output_3d"):
        """
        Return unique project names found in `root`.
        If a filename ends with _<digits> (e.g., "project_3"), strip that suffix.
        File extensions are ignored.
        """
        projects = []
        seen = set()

        if not os.path.isdir(root):
            return projects

        for entry in os.listdir(root):
            # drop extension
            name, _ext = os.path.splitext(entry)

            # strip trailing "_<number>" if present
            parts = name.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                name = parts[0]

            if name and name not in seen:
                seen.add(name)
                projects.append(name)

        return projects

    def _write_manifest(self, name: str, media_paths, mode: str):
        """
        Write output_3d/<name>/project.json with paths *relative* to that folder
        when possible. Falls back to absolute only if a relative path cannot be
        formed (e.g., different drive on Windows).
        """
        from pathlib import Path
        import os, json

        base = Path("output_3d") / name
        base.mkdir(parents=True, exist_ok=True)

        rel_paths = []
        for p in (media_paths or []):
            p_abs = Path(p).resolve()
            try:
                # Prefer OS-level relpath so we can go outside the project tree
                rel = os.path.relpath(str(p_abs), start=str(base))
                rel_paths.append(rel)
            except Exception:
                # Different drive or other OS limitation → keep absolute
                rel_paths.append(str(p_abs))

        manifest = {"mode": mode, "media": rel_paths}
        (base / "project.json").write_text(json.dumps(manifest, indent=2))

    def create_project(self, name, videos, persons):
        """
        Create project that references original video files via manifest.
        Copies only person npz files to output_3d.
        """
        # 1) Write manifest (videos)
        self._write_manifest(name, videos, mode="video")

        # 2) Per-person dirs + npz copies (unchanged behavior)
        for person, file in enumerate(persons):
            person_path = Path("output_3d") / (f"{name}_{person}" if len(persons) > 1 else f"{name}")
            person_path.mkdir(parents=True, exist_ok=True)
            if file is not None:
                src = Path(file)
                dst = person_path / "hand_poses_2d.npz"
                if src.resolve() != dst.resolve():
                    shutil.copy(src, dst)

    def create_project_fotos(self, name, fotos, persons):
        """
        Create project that references original image files via manifest.
        Copies only person npz files to output_3d.
        """
        # 1) Write manifest (fotos)
        self._write_manifest(name, fotos, mode="foto")

        # 2) Per-person dirs + npz copies (unchanged behavior)
        for person, file in enumerate(persons):
            person_path = Path("output_3d") / (f"{name}_{person}" if len(persons) > 1 else f"{name}")
            person_path.mkdir(parents=True, exist_ok=True)
            if file is not None:
                src = Path(file)
                dst = person_path / "hand_poses_2d.npz"
                if src.resolve() != dst.resolve():
                    shutil.copy(src, dst)

    def open_project_specific_frame(self, name, videos, persons, frame_idx: int):
        """
        Create/prepare project assets, then open the project at a specific frame.
        Uses a manifest to reference the original video(s) without copying.
        """
        if not videos or len(videos) != 1:
            raise ValueError("open_project_specific_frame requires exactly one selected video.")
        video_path = Path(videos[0])
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # 1) Write/overwrite manifest to reference the chosen video
        self._write_manifest(name, [str(video_path)], mode="video")

        # 2) Per-person setup (same as before)
        num_persons = len(persons or [])
        for person_idx, npz_file in enumerate(persons or []):
            dir_name = name if num_persons == 1 else f"{name}_{person_idx}"
            out_dir = Path("output_3d") / dir_name
            out_dir.mkdir(parents=True, exist_ok=True)
            if npz_file:
                src = Path(npz_file)
                dst = out_dir / "hand_poses_2d.npz"
                if src.resolve() != dst.resolve():
                    shutil.copy(src, dst)

        # Camera name is the stem of the chosen video
        camera_name = video_path.stem

        # 3) Open project at specific camera+frame
        self.open_project(
            name,
            initial_camera_name=camera_name,
            initial_frame_idx=int(frame_idx),
        )


    def parse_args(self):
        """
        Parses command line arguments for the application.
        """
        parser = argparse.ArgumentParser(description='A tool for manually labeling hand poses')
        parser.add_argument('dataset', type=str, nargs='?', help='Name of the dataset to use', default=None)
        return parser.parse_args()

def main():
    """Starts the application with the dataset cha1 or the dataset specified in the first argument."""
    return App().exec_()

main()
