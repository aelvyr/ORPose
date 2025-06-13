from matplotlib.lines import Line2D
from matplotlib.patches import Circle

class Line(Line2D):
    def __init__(self, start, end, color):
        super().__init__([start.x, end.x], [start.y, end.y], color=color)
        self.start = start
        self.end = end
        self.color = color

class Keypoint(Circle):
    index = 0
    def __init__(self, x, y, index, radius, color, picker, **kwargs):
        if 'picker' not in kwargs and picker:
            kwargs['picker'] = True
        super().__init__((x, y), radius=radius, color=color, **kwargs)
        self.index = index
        self.x = x
        self.y = y
        self.color = color

class Keypoints:
    colors = ['red', 'green', 'blue', 'yellow', 'magenta']
    def keypoint_color(self, index: int):
        if index == 0:
            return self.colors[0]
        index = (index-1) // 4
        return self.colors[index]

    def init_keypoints(self, radius: float):
        self.keypoints = []
        for pos in self.positions:
            keypoint = Keypoint(pos[0], pos[1], len(self.keypoints), color=self.keypoint_color(len(self.keypoints)), radius=radius, picker=self.pickable)
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

    def __init__(self, app, axes, keypoint_positions, render_check = (lambda _: True), radius=5.0, pickable=False):
        self.app = app
        self.axes = axes
        self.positions = keypoint_positions
        self.render_check = render_check
        self.pickable = pickable
        self.init_keypoints(radius)
        return

    def should_keypoint_render(self, keypoint: Keypoint):
        if 0 <= keypoint.index <= len(self.keypoints):
            return self.render_check(keypoint.index)

    def draw(self):
        for keypoint in self.keypoints:
            keypoint.set_center(self.positions[keypoint.index])
            if self.app.current_keypoint == keypoint.index:
                keypoint.set_edgecolor('black')
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
