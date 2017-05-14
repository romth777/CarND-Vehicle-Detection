import numpy as np
from collections import deque


class HeatmapHistory:
    def __init__(self):
        self.number_of_iteration = 5
        self.current = None
        self.history = deque(maxlen=self.number_of_iteration)
        self.mean = None
        self.average_weight = [1.0, 2/3, 4/9, 8/27, 16/81]

    def update(self, heatmap):
        self.current = heatmap
        self.history.append(heatmap)
        if len(self.history) > 1:
            self.mean = np.average(self.history, axis=0, weights=self.average_weight[0:len(self.history)])
        else:
            self.mean = self.current
        return self.mean

    def reset(self):
        self.history.clear()
