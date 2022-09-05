import cv2
from numpy import imag
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

from utils.operation_utils import Operation
from utils.drawing_utils import Draw
from utils.pose_utils.const import POSE, PRESENCE_THRESHOLD, VISIBILITY_THRESHOLD

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class Pose():
    def __init__(self, video_reader) -> None:
        self.video_reader = video_reader
        self.operation = Operation()
        self.pushup_counter = 0
        self.plank_counter = 0
        self.squat_counter = 0
        self.pushups_ang1_tracker = []
        self.pushups_ang4_tracker = []
        self.key_points = None
        self.prev_pose = None
        self.current_pose = None
        self.pose_tracker = []
        self.width = int(self.video_reader.get_frame_width())
        self.height = int(self.video_reader.get_frame_height())
        self.video_fps = self.video_reader.get_video_fps()
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.draw = Draw(self.width, self.height)

    def pose_algorithm(self):
        raise NotImplementedError("Requires Subclass implementation.")

    def measure(self):
        raise NotImplementedError("Requires Subclass implementation.")

    def get_keypoints(self, image, pose_result):
        key_points = {}
        image_rows, image_cols, _ = image.shape
        for idx, landmark in enumerate(pose_result.pose_landmarks.landmark):
            if ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                continue
            landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
            if landmark_px:
                key_points[idx] = landmark_px
        return key_points

    def is_point_in_keypoints(self, str_point):
        return POSE[str_point] in self.key_points

    def get_point(self, str_point):
        return self.key_points[POSE[str_point]] if self.is_point_in_keypoints(str_point) else None
    
    def two_line_angle(self, str_point1, str_point2, str_point3):
        coord1 = self.get_point(str_point1)
        coord2 = self.get_point(str_point2)
        coord3 = self.get_point(str_point3)
        return self.operation.angle(coord1, coord2, coord3)

    def one_line_angle(self, str_point1, str_point2):
        coord1 = self.get_point(str_point1)
        coord2 = self.get_point(str_point2)
        return self.operation.angle_of_singleline(coord1, coord2)

    def predict_pose(self):
        ang1 = ang2 = ang3 = ang4 = None
        is_pushup = False
        is_plank = False

        # Calculate angle between lines shoulder-elbow, elbow-wrist
        if self.is_point_in_keypoints("left_shoulder") and self.is_point_in_keypoints("left_elbow") and self.is_point_in_keypoints("left_wrist"):
            ang1 = self.two_line_angle("left_shoulder", "left_elbow", "left_wrist")
        elif self.is_point_in_keypoints("right_shoulder") and self.is_point_in_keypoints("right_elbow") and self.is_point_in_keypoints("right_wrist"):
            ang1 = self.two_line_angle("right_shoulder", "right_elbow", "right_wrist")
        else:
            pass

        # Calculate angle between lines shoulder-hip, hip-ankle
        if self.is_point_in_keypoints("left_shoulder") and self.is_point_in_keypoints("left_hip") and self.is_point_in_keypoints("left_ankle"):
            ang2 = self.two_line_angle("left_shoulder", "left_hip", "left_ankle")
        elif self.is_point_in_keypoints("right_shoulder") and self.is_point_in_keypoints("right_hip") and self.is_point_in_keypoints("right_ankle"):
            ang2 = self.two_line_angle("right_shoulder", "right_hip", "right_ankle")
        else:
            pass

        # Calculate angle of line shoulder-ankle or hip-ankle
        left_shoulder_ankle = self.is_point_in_keypoints("left_shoulder") and self.is_point_in_keypoints("left_ankle")
        right_shoulder_ankle = self.is_point_in_keypoints("right_shoulder") and self.is_point_in_keypoints("right_ankle")
        left_hip_ankle = self.is_point_in_keypoints("left_hip") and self.is_point_in_keypoints("left_ankle")
        right_hip_ankle = self.is_point_in_keypoints("right_hip") and self.is_point_in_keypoints("right_ankle")
        if left_shoulder_ankle or right_shoulder_ankle:
            shoulder = "left_shoulder" if left_shoulder_ankle else "right_shoulder"
            ankle = "left_ankle" if left_shoulder_ankle else "right_ankle"
            ang3 = self.one_line_angle(shoulder, ankle)
        elif left_hip_ankle or right_hip_ankle:
            hip = "left_hip" if left_hip_ankle else "right_hip"
            ankle = "left_ankle" if left_hip_ankle else "right_ankle"
            ang3 = self.one_line_angle(hip, ankle)
        else:
            pass

        # Calculate angle of line elbow-wrist
        left_elbow_wrist = self.is_point_in_keypoints("left_elbow") and self.is_point_in_keypoints("left_wrist")
        right_elbow_wrist = self.is_point_in_keypoints("right_elbow") and self.is_point_in_keypoints("right_wrist")
        if left_elbow_wrist or right_elbow_wrist:
            elbow = "left_elbow" if left_elbow_wrist else "right_elbow"
            wrist = "left_wrist" if left_elbow_wrist else "right_wrist"
            ang4 = self.one_line_angle(elbow, wrist)
        else:
            pass

        if ang3 is not None and ((0 <= ang3 <= 50) or (130 <= ang3 <= 180)):
            if (ang1 is not None or ang2 is not None) and ang4 is not None:
                if (160 <= ang2 <= 180) or (0 <= ang2 <= 20):
                    self.pushup_counter += 1
                    self.pushups_ang1_tracker.append(ang1)
                    self.pushups_ang4_tracker.append(ang4)

        if self.pushup_counter >= 24 and len(self.pushups_ang1_tracker) == 24 and len(self.pushups_ang4_tracker) == 24:
            ang1_diff1 = abs(self.pushups_ang1_tracker[0] - self.pushups_ang1_tracker[12])
            ang1_diff2 = abs(self.pushups_ang1_tracker[12] - self.pushups_ang1_tracker[23])
            ang1_diff_mean = (ang1_diff1 + ang1_diff2) / 2
            ang4_mean = sum(self.pushups_ang4_tracker) / len(self.pushups_ang4_tracker)
            del self.pushups_ang1_tracker[0]
            del self.pushups_ang4_tracker[0]
            if ang1_diff_mean < 5 and not 75 <= ang4_mean <= 105:
                is_plank = True
                is_pushup = False
            else:
                is_pushup = True
                is_plank = False

        if is_pushup:
            return "Pushup"
        elif is_plank:
            return "Plank"

        return None

    def estimate(self) -> None:
        if self.video_reader.is_opened() == False:
            print("Error File Not Found.")

        out = cv2.VideoWriter(f"output.avi", self.fourcc, self.video_fps, (self.width, self.height))
        while self.video_reader.is_opened():
            image = self.video_reader.read_frame()
            if image is None:
                print("Ignoring empty camera frame.")
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = self.draw.overlay(image)
            image = self.draw.skeleton(image, results)

            if results.pose_landmarks is not None:
                self.key_points = self.get_keypoints(image, results)
                estimated_pose = self.predict_pose()
                if estimated_pose is not None:
                    self.current_pose = estimated_pose
                    self.pose_tracker.append(self.current_pose)
                    if len(self.pose_tracker) == 10 and len(set(self.pose_tracker[-6:])) == 1:
                        image = self.draw.pose_text(image, "Prediction: " + estimated_pose)

            if len(self.pose_tracker) == 10:
                del self.pose_tracker[0]
                self.prev_pose = self.pose_tracker[-1]

            out.write(image)
            cv2.imshow('Estimation of Exercise', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.video_reader.release()


class Pushup(Pose):
    def __init__(self, video_reader) -> None:
        super().__init__(video_reader)
        self.video_reader = video_reader
        self.pushups_count = 0

    def _draw(self, image):
        left_shoulder_wrist_foot = self.is_point_in_keypoints("left_shoulder") and self.is_point_in_keypoints("left_wrist") and self.is_point_in_keypoints("left_foot_index")
        right_shoulder_wrist_foot = self.is_point_in_keypoints("right_shoulder") and self.is_point_in_keypoints("right_wrist") and self.is_point_in_keypoints("right_foot_index")
        if left_shoulder_wrist_foot:
            image = self.draw.draw_line(image, self.get_point("left_shoulder"), self.get_point("left_wrist"))
            image = self.draw.draw_line(image, self.get_point("left_wrist"), self.get_point("left_foot_index"))
            image = self.draw.draw_line(image, self.get_point("left_foot_index"), self.get_point("left_shoulder"))
        elif right_shoulder_wrist_foot:
            image = self.draw.draw_line(image, self.get_point("right_shoulder"), self.get_point("right_wrist"))
            image = self.draw.draw_line(image, self.get_point("right_wrist"), self.get_point("right_foot_index"))
            image = self.draw.draw_line(image, self.get_point("right_foot_index"), self.get_point("right_shoulder"))
        else:
            pass
        return image

    def pose_algorithm(self):
        pass

    def measure(self) -> None:
        if self.video_reader.is_opened() == False:
            print("Error File Not Found.")

        out = cv2.VideoWriter(f"output.avi", self.fourcc, self.video_fps, (self.width, self.height))
        while self.video_reader.is_opened():
            image = self.video_reader.read_frame()
            if image is None:
                print("Ignoring empty camera frame.")
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = self.draw.overlay(image)
            # image = self.draw.skeleton(image, results)

            if results.pose_landmarks is not None:
                self.key_points = self.get_keypoints(image, results)
                self.pose_algorithm()
                image = self._draw(image)
                image = self.draw.pose_text(image, "Pushups Count: " + str(self.pushups_count))

            out.write(image)
            cv2.imshow('Pushups', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.video_reader.release() 