"""
This module contains code to render keypoints and the lines connecting them using matplotlib.
"""

from matplotlib.lines import Line2D
from matplotlib.patches import Circle

class Keypoint(Circle):
    """
    Wrapper around matplotlib Circle used to apply default properties
    and store the additional information in the artist itself.

    It contains the following attributes:
        - index: The index of the keypoint.
        - x: The x-coordinate of the keypoint.
        - y: The y-coordinate of the keypoint.
        - color: The color of the keypoint.
    """
    index = 0
    def __init__(self, x, y, index, radius, color, picker):
        """
        Initialize a Keypoint object.

        Args:
            x (float): The x-coordinate of the keypoint.
            y (float): The y-coordinate of the keypoint.
            index (int): The index of the keypoint.
            radius (float): The radius of the keypoint.
            color (str): The color of the keypoint.
            picker (bool): Whether the keypoint is pickable.
        """
        super().__init__((x, y), radius=radius, color=color, picker=picker, visible=False, linewidth=3.0)
        self.index = index
        self.x = x
        self.y = y
        self.color = color

class Line(Line2D):
    """
    Wrapper around matplotlib Line2D used to apply default properties
    and store the additional information in the artist itself.

    It contains the following attributes:
        - start: A reference to the keypoint defining the start of the line.
        - end: A reference to the keypoint defining the end of the line.
    """
    def __init__(self, start: Keypoint, end: Keypoint, color: str):
        """
        Initialize a Line object.

        Args:
            start (Keypoint): The start point of the line.
            end (Keypoint): The end point of the line.
            color (str): The color of the line.
        """
        super().__init__([start.x, end.x], [start.y, end.y], color=color, visible=False)
        self.start = start
        self.end = end

class Keypoints:
    """
    This class is used to draw keypoints on matplotlib axes

    It contains the following attributes:
        - colors: A list of colors for the keypoints used to determine the color of each keypoint based on its index.
        - app: A reference to the application instance.
        - axes: The matplotlib axes where the keypoints will be drawn.
        - positions: The positions at which the keypoints will be drawn.
        - render_check: A function which determines whether the keypoints should be rendered.
        - keypoints: A list of Keypoint artists.
        - keypoint_lines: A list of line artists.
    """
    colors = ['red', 'green', 'blue', 'yellow', 'magenta']

    def __init__(self, project, axes, positions, radius=5.0, pickable=False):
        """
        Initializes the Keypoints object.

        Args:
            project: A reference to the application instance.
            axes: The matplotlib axes where the keypoints will be drawn.
            positions: The positions at which the keypoints will be drawn.
            render_check: A function which determines whether the keypoints should be rendered.
            radius: The radius of the keypoints.
            pickable: Determines whether the keypoints are pickable.
        """
        self.project = project
        self.axes = axes
        self.positions = positions
        self.init_keypoints(radius, pickable)

    def keypoint_color(self, index: int):
        """
        Returns the color of the keypoint with the given index.

        Args:
            index: The index of the keypoint.

        Returns:
            The color of the keypoint.
        """
        if index == 0:
            return self.colors[0]
        index = (index-1) // 4
        return self.colors[index]

    def init_keypoints(self, radius: float, pickable: bool):
        """
        Initializes the keypoint artists and the line artists.
        """
        self.keypoints = []
        for pos in self.positions:
            keypoint = Keypoint(pos[0], pos[1], len(self.keypoints), color=self.keypoint_color(len(self.keypoints)), radius=radius, picker=pickable)
            self.axes.add_patch(keypoint)
            self.keypoints.append(keypoint)
        self.keypoint_lines = []
        for end in self.keypoints:
            if end.index == 0:
                continue
            prev_index = end.index - 1
            if end.index % 4 == 1:
                prev_index = 0
            start = self.keypoints[prev_index]
            line = Line(start, end, color=end.color, )
            self.axes.add_line(line)
            self.keypoint_lines.append(line)

    def should_keypoint_render(self, keypoint: Keypoint):
        """
        Determines whether a keypoint should be rendered.
        """
        return 0 <= keypoint.index <= len(self.keypoints)

    def set_radius(self, radius: float):
        for keypoint in self.keypoints:
            keypoint.set_radius(radius)

    def draw(self):
        """
        Draws the keypoints and their connections.
        """
        print("Drawing keypoints")
        for keypoint in self.keypoints:
            keypoint.set_center(self.positions[keypoint.index])
            if self.project.current_keypoint == keypoint.index:
                edge_color = 'white' if self.project.dark_mode else 'black'
                keypoint.set_edgecolor(edge_color)
            else:
                keypoint.set_edgecolor(keypoint.color)
            if self.should_keypoint_render(keypoint):
                keypoint.set_visible(True)
            else:
                keypoint.set_visible(False)
        for line in self.keypoint_lines:
            start = self.positions[line.start.index]
            end = self.positions[line.end.index]
            line.set_data([start[0], end[0]], [start[1], end[1]])
            if self.should_keypoint_render(line.start) and self.should_keypoint_render(line.end):
                line.set_visible(True)
            else:
                line.set_visible(False)
        self.axes.figure.canvas.draw()
