import argparse
from project import Project
from PyQt5.QtWidgets import QApplication, QErrorMessage
from ui.launcher import Launcher
import sys
import os
from pathlib import Path
import shutil

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

    def open_project(self, dataset_name):
        """
        Opens the project specified by the dataset name.
        If opening fails, it displays an error message.

        Args:
            dataset_name (str): The name of the dataset to open.
        """
        try:
            self.current_project = Project(self, dataset_name)
            self.current_project.window.show()
            print("Project window shown")
        except ValueError as e:
            print(e)
            QErrorMessage(self, f"Error opening project: {e}").exec_()
            quit(-1)

    def get_project_names(self):
        """
        Returns a list of project names which the application knows about.
        """
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
            shutil.copy(file, inputs / file.name)
        for person, file in enumerate(persons):
            person_path = Path("output_3d") / f"{name}_{person}"
            person_path.mkdir(parents=True, exist_ok=True)
            if file is not None:
                shutil.copy(file, person_path / "hand_poses_2d.npz")

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
