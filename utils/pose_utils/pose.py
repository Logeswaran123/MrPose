import cv2
from numpy import imag
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from utils.pose_utils.operation_utils import Operation

from ..drawing_utils import Draw
from .const import POSE, PRESENCE_THRESHOLD, VISIBILITY_THRESHOLD
from .operation_utils import Operation

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class Pose():
    def __init__(self) -> None:
        self.operation = Operation()

    def pose_algorithm(self):
        raise NotImplementedError("Requires Subclass implementation.")

    def estimate(self):
        raise NotImplementedError("Requires Subclass implementation.")


class Pushup(Pose):
    def __init__(self, video_reader) -> None:
        super().__init__()
        self.video_reader = video_reader
        self.prev_pose = None
        self.current_pose = None
        self.pose_tracker = []

    def get_keypoints(self, image, pose_result):
        key_points = {}
        for idx, landmark in enumerate(pose_result.pose_landmarks.landmark):
            if ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                continue
            landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                        image.shape[0], image.shape[1])
            if landmark_px:
                key_points[idx] = landmark_px
        return key_points
    
    def pose_algorithm(self, key_points):
        ang1 = ang2 = None
        is_pushup = False

        # Calculate angle between lines shoulder-elbow, elbow-wrist
        if POSE["left_shoulder"] in key_points and POSE["left_elbow"] in key_points and POSE["left_wrist"] in key_points:
            left_shoulder = key_points[POSE["left_shoulder"]]
            left_elbow = key_points[POSE["left_elbow"]]
            left_wrist = key_points[POSE["left_wrist"]]
            ang1 = self.operation.angle(left_shoulder, left_elbow, left_wrist)
        elif POSE["right_shoulder"] in key_points and POSE["right_elbow"] in key_points and POSE["right_wrist"] in key_points:
            right_shoulder = key_points[POSE["right_shoulder"]]
            right_elbow = key_points[POSE["right_elbow"]]
            right_wrist = key_points[POSE["right_wrist"]]
            ang1 = self.operation.angle(right_shoulder, right_elbow, right_wrist)
        else:
            pass

        # Calculate angle between lines shoulder-hip, hip-ankle
        if POSE["left_shoulder"] in key_points and POSE["left_hip"] in key_points and POSE["left_ankle"] in key_points:
            left_shoulder = key_points[POSE["left_shoulder"]]
            left_hip = key_points[POSE["left_hip"]]
            left_ankle = key_points[POSE["left_ankle"]]
            ang2 = self.operation.angle(left_shoulder, left_hip, left_ankle)
        elif POSE["right_shoulder"] in key_points and POSE["right_hip"] in key_points and POSE["right_ankle"] in key_points:
            right_shoulder = key_points[POSE["right_shoulder"]]
            right_hip = key_points[POSE["right_hip"]]
            right_ankle = key_points[POSE["right_ankle"]]
            ang2 = self.operation.angle(right_shoulder, right_elbow, right_ankle)
        else:
            pass

        # Calculate angle of line shoulder-ankle or hip-ankle
        left_shoulder_ankle = POSE["left_shoulder"] in key_points and POSE["left_ankle"] in key_points
        right_shoulder_ankle = POSE["right_shoulder"] in key_points and POSE["right_ankle"] in key_points
        left_hip_ankle = POSE["left_hip"] in key_points and POSE["left_ankle"] in key_points
        right_hip_ankle = POSE["right_hip"] in key_points and POSE["right_ankle"] in key_points
        if left_shoulder_ankle or right_shoulder_ankle:
            shoulder = key_points[POSE["left_shoulder"]] if left_shoulder_ankle else key_points[POSE["right_shoulder"]]
            ankle = key_points[POSE["left_ankle"]] if left_shoulder_ankle else key_points[POSE["right_ankle"]]
            ang3 = self.operation.angle_of_singleline(shoulder, ankle)
        elif left_hip_ankle or right_hip_ankle:
            hip = key_points[POSE["left_hip"]] if left_hip_ankle else key_points[POSE["right_hip"]]
            ankle = key_points[POSE["left_ankle"]] if left_hip_ankle else key_points[POSE["right_ankle"]]
            ang3 = self.operation.angle_of_singleline(hip, ankle)
        else:
            pass

        if ang3 is not None and ang3 <= 50:
            if ang1 is not None or ang2 is not None:
                if (160 <= ang2 <= 180) or (0 <= ang2 <= 20):
                    is_pushup = True

        if is_pushup:
            return "Pushup"

        return None


    def estimate(self) -> None:
        if self.video_reader.is_opened() == False:
            print("Error File Not Found.")

        width = int(self.video_reader.get_frame_width())
        height = int(self.video_reader.get_frame_height())
        channels = 3
        video_fps = self.video_reader.get_video_fps()

        total_frames = self.video_reader.get_total_frames()

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(f"output.avi", fourcc, video_fps, (width, height))

        draw = Draw(width, height)

        frame_counter = 0
        while self.video_reader.is_opened():
            image = self.video_reader.read_frame()
            if image is None:
                print("Ignoring empty camera frame.")
                break

            frame_counter += 1
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = draw.overlay(image)
            image = draw.skeleton(image, results)

            if results.pose_landmarks is not None:
                key_points = self.get_keypoints(image, results)
                estimated_pose = self.pose_algorithm(key_points)
                if estimated_pose is not None:
                    self.current_pose = estimated_pose
                    self.pose_tracker.append(self.current_pose)
                    if len(self.pose_tracker) == 10 and len(set(self.pose_tracker[-6:])) == 1:
                        image = draw.pose_text(image, estimated_pose)

            if len(self.pose_tracker) == 10:
                del self.pose_tracker[0]
                self.prev_pose = self.pose_tracker[-1]

            out.write(image)
            cv2.imshow('Estimation of Exercise', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.video_reader.release() 