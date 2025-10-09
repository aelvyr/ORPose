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
        dataset_name = self.parse_args().dataset
        if dataset_name is not None:
            self.open_project(dataset_name)
        else:
            self.launcher = Launcher(self)
            print("Launcher created")
            self.launcher.show()
            print("Launcher shown")

    def open_project(self, dataset_name, initial_camera_name=None, initial_frame_idx=None):
        """
        Open the project UI.

        Args:
            dataset_name (str): dataset/folder name under inputs/
            initial_camera_name (str|None): optional camera to select on load
            initial_frame_idx (int|None): optional exact frame index to jump to
        """
        print("Dataset name:", dataset_name)
        try:
            self.current_project = Project(
                self,
                dataset_name,
                initial_camera_name=initial_camera_name,
                initial_frame_idx=initial_frame_idx,
            )
            self.current_project.window.show()
            print("Project window shown")
        except Exception as e:
            print(f"[open_project] Failed to open project '{dataset_name}': {e}")
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

    def create_project(self, name, videos, persons):
        """
        Creates a new project with the given name, videos, and persons.

        Args:
            name (str): The name of the project.
            videos (list): A list of video files.
            persons (list): A list of person files.
        """
        inputs = Path("inputs") / name
        inputs.mkdir(parents=True, exist_ok=True)
        for file in videos:
            file = Path(file)
            if file.resolve() == (inputs / file.name).resolve():
                continue
            shutil.copy(file, inputs / file.name)
        for person, file in enumerate(persons):
            person_path = Path("output_3d") / f"{name}_{person}" if len(persons) > 1 else Path("output_3d") / f"{name}"
            person_path.mkdir(parents=True, exist_ok=True)
            if file is not None:
                if file.resolve() == (person_path / "hand_poses_2d.npz").resolve():
                    continue
                shutil.copy(file, person_path / "hand_poses_2d.npz")

    def create_project_fotos(self, name, fotos, persons):
        """
        Creates a new project with the given name, videos, and persons.

        Args:
            name (str): The name of the project.
            fotos (list): A list of foto files.
            persons (list): A list of person files.
        """
        inputs = Path("inputs") / name
        inputs.mkdir(parents=True, exist_ok=True)
        for file in fotos:
            file = Path(file)
            if file.resolve() == (inputs / file.name).resolve():
                continue
            shutil.copy(file, inputs / file.name)
        for person, file in enumerate(persons):
            person_path = Path("output_3d") / f"{name}_{person}" if len(persons) > 1 else Path("output_3d") / f"{name}"
            person_path.mkdir(parents=True, exist_ok=True)
            if file is not None:
                if file.resolve() == (person_path / "hand_poses_2d.npz").resolve():
                    continue
                shutil.copy(file, person_path / "hand_poses_2d.npz")

    def open_project_specific_frame(self, name, videos, persons, frame_idx: int):
        """
        Create/prepare project assets (like other flows), then open the project
        at a specific frame of the single selected video.
        """
        # --- validation ---
        if not videos or len(videos) != 1:
            raise ValueError("open_project_specific_frame requires exactly one selected video.")
        video_path = Path(videos[0])
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # --- copy inputs (same behavior as other flows) ---
        inputs_dir = Path("inputs") / name
        inputs_dir.mkdir(parents=True, exist_ok=True)
        dst_video = inputs_dir / video_path.name
        if video_path.resolve() != dst_video.resolve():
            shutil.copy(video_path, dst_video)

        # --- per-person setup (same behavior as other flows) ---
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

        # Determine camera name (video-mode cameras are basenames without extension)
        camera_name = video_path.stem

        # --- open the project at the specific camera+frame ---
        # If your existing open_project already builds and shows the UI,
        # pass the initial camera/frame in (after updating Project to accept these).
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
