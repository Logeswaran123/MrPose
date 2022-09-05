import argparse

from utils.exercise_utils import Exercise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', "--video", required=True, help="Path to video source file", type=str)
    parser.add_argument('-e', "--exercise", required=False, default="predict",
                        help="Type of exercise in video source",
                        type=str, choices=['predict', 'pushup', 'plank'])
    args = parser.parse_args()
    video = args.video
    exercise = args.exercise
    pose = Exercise(video, exercise)
    pose.estimate_exercise()