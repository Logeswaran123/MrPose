import sys

from utils.pose_utils.pose import Pose, Pushup, Squat
from utils.video_reader_utils import VideoReader

class Exercise():
    def __init__(self, filename: str, exercise: str) -> None:
        self.video_reader = VideoReader(filename)
        if exercise == "predict":
            exercise = "pose"
        self.exercise = exercise.lower().capitalize()

    def estimate_exercise(self):
        pose_estimator = getattr(sys.modules[__name__], self.exercise)
        pose_estimator = pose_estimator(self.video_reader)
        pose_estimator.estimate() if self.exercise == "Pose" else pose_estimator.measure()
