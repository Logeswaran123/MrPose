import time
import datetime

class Timer:
    """ Helper class for timer utilities """
    def __init__(self):
        self._current_time = 0.0
        self._start_time = 0.0
        self._end_time = 0.0
        self._total_time = 0.0
        self._counter = 0.0

    def start(self, start_time=None):
        """ Strat timer """
        if start_time is not None:
            assert type(start_time) in [int, float]
            self._current_time = start_time
        else:
            self._current_time = time.time()
        self._start_time = self._current_time

    def end(self, end_time=None):
        """ End timer """
        if end_time is not None:
            assert type(end_time) in [int, float]
            self._current_time = end_time - self._current_time
            self._total_time += self._current_time
        else:
            self._current_time = time.time() - self._current_time
            self._total_time += self._current_time
            self._counter += 1
        self._start_time = 0.0
        self._end_time = self._current_time

    def reset(self):
        """ Reset timer """
        self._current_time = 0.0
        self._total_time = 0.0
        self._counter = 0.0

    def get_current_time(self):
        """ Get current time """
        return time.time() - self._start_time if self._start_time > 0.0 else self._start_time

    def get_average_time(self):
        """ Get average time """
        avg_time = 0
        try:
            avg_time = self._total_time/self._counter
            return avg_time
        except ZeroDivisionError:
            return 0

    def get_current_fps(self):
        """ Get current frames per second """
        fps = 0
        try:
            fps = 1/self._current_time
            return fps
        except ZeroDivisionError:
            return 0

    def get_average_fps(self):
        """ Get average frames per second """
        avg_fps = 0
        try:
            avg_fps = 1/(self._total_time/self._counter)
            return avg_fps
        except ZeroDivisionError:
            return 0

    def convert_time(self, time_in_sec):
        """ Convert seconds to readable format """
        return datetime.timedelta(seconds=time_in_sec)
