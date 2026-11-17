# Ported from r1-a7-autonomous-checkout robot/xr-teleoperate/teleop/utils/weighted_moving_filter.py
# (visualization code removed; filter logic unchanged)
import numpy as np


class WeightedMovingFilter:
    def __init__(self, weights, data_size=14, skip_duplicates=True):
        self._window_size = len(weights)
        self._weights = np.array(weights)
        assert np.isclose(np.sum(self._weights), 1.0), "[WeightedMovingFilter] the sum of weights list must be 1.0!"
        self._data_size = data_size
        self._skip_duplicates = bool(skip_duplicates)
        self._filtered_data = np.zeros(self._data_size)
        self._data_queue = []

    def _apply_filter(self):
        if len(self._data_queue) < self._window_size:
            return self._data_queue[-1]

        data_array = np.array(self._data_queue)
        return np.sum(data_array * self._weights[::-1, None], axis=0)

    def add_data(self, new_data):
        assert len(new_data) == self._data_size

        if (
            self._skip_duplicates
            and len(self._data_queue) > 0
            and np.array_equal(new_data, self._data_queue[-1])
        ):
            return  # skip duplicate data

        if len(self._data_queue) >= self._window_size:
            self._data_queue.pop(0)

        self._data_queue.append(new_data)
        self._filtered_data = self._apply_filter()

    @property
    def filtered_data(self):
        return self._filtered_data
