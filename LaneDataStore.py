LEARNING_RATE = 0.5

"""
This class stored information to be carried between frames in video
"""


class LaneDataStore:
    """Object to store lane data between each frame"""

    def __init__(self):
        self.left_fit_coef = None
        self.right_fit_coef = None
        self.radius = None
        self.pos = None

        self.mtx = None
        self.dist = None
        self.M = None
        self.Minv = None

    def reset(self):
        self.left_fit_coef = None
        self.right_fit_coef = None
        self.radius = None
        self.pos = None

    def is_initialized(self):
        return self.left_fit_coef is not None and self.right_fit_coef is not None

    def update(self, new_left_fit_coef, new_right_fit_coef, new_radius, new_pos):
        if self.left_fit_coef is None:
            self.left_fit_coef = new_left_fit_coef
        else:
            a = LEARNING_RATE * new_left_fit_coef[0] + (1 - LEARNING_RATE) * self.left_fit_coef[0]
            b = LEARNING_RATE * new_left_fit_coef[1] + (1 - LEARNING_RATE) * self.left_fit_coef[1]
            c = LEARNING_RATE * new_left_fit_coef[2] + (1 - LEARNING_RATE) * self.left_fit_coef[2]
            self.left_fit_coef = [a, b, c]

        if self.right_fit_coef is None:
            self.right_fit_coef = new_right_fit_coef
        else:
            a = LEARNING_RATE * new_right_fit_coef[0] + (1 - LEARNING_RATE) * self.right_fit_coef[0]
            b = LEARNING_RATE * new_right_fit_coef[1] + (1 - LEARNING_RATE) * self.right_fit_coef[1]
            c = LEARNING_RATE * new_right_fit_coef[2] + (1 - LEARNING_RATE) * self.right_fit_coef[2]
            self.right_fit_coef = [a, b, c]

        if self.radius is None:
            self.radius = new_radius
        else:
            self.radius = LEARNING_RATE * new_radius + (1 - LEARNING_RATE) * self.radius

        if self.pos is None:
            self.pos = new_pos
        else:
            self.pos = LEARNING_RATE * new_pos + (1 - LEARNING_RATE) * self.pos
