import argparse
from project import Project
from PyQt5.QtWidgets import QApplication
from ui import ProjectWindow
from ui.launcher import Launcher
import sys
import os
from pathlib import Path
import shutil

class App(QApplication):
    def __init__(self):
        super().__init__(sys.argv)
        dataset_name = self.parse_args().dataset
        if dataset_name is not None:
            self.open_project(dataset_name)
        else:
            self.launcher = Launcher(self)
            print("Launcher created")
            self.launcher.show()
            print("Launcher shown")

    def open_project(self, dataset_name):
        self.current_project = Project(self, dataset_name)
        self.current_project.window.show()
        print("Project window shown")

    def get_project_names(self):
        projects = []
        for file in os.listdir("output_3d"):
            parts = file.split("_")
            if len(parts) < 2:
                projects.append(parts[0])
                continue
            name = file[:-(len(parts[-1])+1)]
            if name not in projects:
                projects.append(name)
        return projects

    def create_project(self, name, videos, persons):
        inputs = Path("inputs") / name
        inputs.mkdir(parents=True, exist_ok=True)
        for file in videos:
            file = Path(file)
            shutil.copy(file, inputs / file.name)
        for person, file in enumerate(persons):
            person_path = Path("output_3d") / f"{name}_{person}"
            person_path.mkdir(parents=True, exist_ok=True)
            if file is not None:
                shutil.copy(file, person_path / "hand_poses_2d.npz")

    def parse_args(self):
        parser = argparse.ArgumentParser(description='A tool for manually labeling hand poses')
        parser.add_argument('dataset', type=str, nargs='?', help='Name of the dataset to use', default=None)
        return parser.parse_args()

def main():
    """Starts the application with the dataset cha1 or the dataset specified in the first argument."""
    return App().exec_()

main()
