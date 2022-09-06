import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import random

from utils.operation_utils import Operation
from utils.timer_utils import Timer
from utils.drawing_utils import Draw
from utils.pose_utils.const import POSE, PRESENCE_THRESHOLD, VISIBILITY_THRESHOLD

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class Pose():
    """ Base: Pose Class """
    def __init__(self, video_reader) -> None:
        self.video_reader = video_reader
        self.operation = Operation()
        self.pushup_counter = self.plank_counter = self.squat_counter = 0
        self.key_points = self.prev_pose = self.current_pose = None
        self.ang1_tracker = []
        self.ang4_tracker = []
        self.pose_tracker = []
        self.headpoint_tracker = []
        self.width = int(self.video_reader.get_frame_width())
        self.height = int(self.video_reader.get_frame_height())
        self.video_fps = self.video_reader.get_video_fps()
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.draw = Draw(self.width, self.height)

    def pose_algorithm(self):
        """ Pose subclass algorithm """
        raise NotImplementedError("Requires Subclass implementation.")

    def measure(self):
        """ Pose subclass measure pose """
        raise NotImplementedError("Requires Subclass implementation.")

    def get_keypoints(self, image, pose_result):
        """ Get keypoints """
        key_points = {}
        image_rows, image_cols, _ = image.shape
        for idx, landmark in enumerate(pose_result.pose_landmarks.landmark):
            if ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                continue
            landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                            image_cols, image_rows)
            if landmark_px:
                key_points[idx] = landmark_px
        return key_points

    def is_point_in_keypoints(self, str_point):
        """ Check if point is in keypoints """
        return POSE[str_point] in self.key_points

    def get_point(self, str_point):
        """ Get point from keypoints """
        return self.key_points[POSE[str_point]] if self.is_point_in_keypoints(str_point) else None

    def get_available_point(self, points):
        """
        Get highest priority keypoint from points list.
        i.e. first index is 1st priority, second index is 2nd priority, and so on.
        """
        available_point = None
        for point in points:
            if self.is_point_in_keypoints(point) and available_point is None:
                available_point = self.get_point(point)
                break
        return available_point

    def two_line_angle(self, str_point1, str_point2, str_point3):
        """ Angle between two lines """
        coord1 = self.get_point(str_point1)
        coord2 = self.get_point(str_point2)
        coord3 = self.get_point(str_point3)
        return self.operation.angle(coord1, coord2, coord3)

    def one_line_angle(self, str_point1, str_point2):
        """ Angle of a line """
        coord1 = self.get_point(str_point1)
        coord2 = self.get_point(str_point2)
        return self.operation.angle_of_singleline(coord1, coord2)

    def predict_pose(self):
        """ Predict pose """
        ang1 = ang2 = ang3 = ang4 = None
        is_pushup = is_plank = is_squat = False

        # Calculate angle between lines shoulder-elbow, elbow-wrist
        if self.is_point_in_keypoints("left_shoulder") and \
            self.is_point_in_keypoints("left_elbow") and \
            self.is_point_in_keypoints("left_wrist"):
            ang1 = self.two_line_angle("left_shoulder", "left_elbow", "left_wrist")
        elif self.is_point_in_keypoints("right_shoulder") and \
            self.is_point_in_keypoints("right_elbow") and \
            self.is_point_in_keypoints("right_wrist"):
            ang1 = self.two_line_angle("right_shoulder", "right_elbow", "right_wrist")
        else:
            pass

        # Calculate angle between lines shoulder-hip, hip-ankle
        if self.is_point_in_keypoints("left_shoulder") and \
            self.is_point_in_keypoints("left_hip") and \
            self.is_point_in_keypoints("left_ankle"):
            ang2 = self.two_line_angle("left_shoulder", "left_hip", "left_ankle")
        elif self.is_point_in_keypoints("right_shoulder") and \
            self.is_point_in_keypoints("right_hip") and \
            self.is_point_in_keypoints("right_ankle"):
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

        left_knee_ankle = self.is_point_in_keypoints("left_knee") and self.is_point_in_keypoints("left_ankle")
        right_knee_ankle = self.is_point_in_keypoints("right_knee") and self.is_point_in_keypoints("right_ankle")
        if left_knee_ankle or right_knee_ankle:
            knee = "left_knee" if left_knee_ankle else "right_knee"
            ankle = "left_ankle" if left_knee_ankle else "right_ankle"
            ang5 = self.one_line_angle(knee, ankle)
        else:
            pass

        if ang3 is not None and ((0 <= ang3 <= 50) or (130 <= ang3 <= 180)):
            if (ang1 is not None or ang2 is not None) and ang4 is not None:
                if (160 <= ang2 <= 180) or (0 <= ang2 <= 20):
                    self.pushup_counter += 1
                    self.ang1_tracker.append(ang1)
                    self.ang4_tracker.append(ang4)

        if self.pushup_counter >= 24 and len(self.ang1_tracker) == 24 and len(self.ang4_tracker) == 24:
            ang1_diff1 = abs(self.ang1_tracker[0] - self.ang1_tracker[12])
            ang1_diff2 = abs(self.ang1_tracker[12] - self.ang1_tracker[23])
            ang1_diff_mean = (ang1_diff1 + ang1_diff2) / 2
            ang4_mean = sum(self.ang4_tracker) / len(self.ang4_tracker)
            del self.ang1_tracker[0]
            del self.ang4_tracker[0]
            if ang1_diff_mean < 5 and not 75 <= ang4_mean <= 105:
                is_plank = True
                is_pushup = is_squat = False
            else:
                is_pushup = True
                is_plank = is_squat = False

        # Distance algorithm
        head_point = self.get_available_point(["nose", "left_ear", "right_ear", "left_eye", "right_eye"])
        hip = self.get_available_point(["left_hip", "right_hip"])
        foot = self.get_available_point(["left_foot_index", "right_foot_index", "left_heel", "right_heel", "left_ankle", "right_ankle"])
        self.headpoint_tracker.append(head_point[1]) # height only
        if ang3 is not None and ang5 is not None:
            if ((70 <= ang3 <= 110) or (70 <= ang5 <= 110)) and len(self.headpoint_tracker) == 24:
                height_mean = int(sum(self.headpoint_tracker) / len(self.headpoint_tracker))
                height_norm = self.operation.normalize(height_mean, head_point[1], foot[1])
                del self.headpoint_tracker[0]
                if height_norm < 0:
                    is_squat = True
                    is_pushup = is_plank = False
                else:
                    is_squat = False
        
        if len(self.ang1_tracker) == 24:
            del self.ang1_tracker[0]
        if len(self.ang4_tracker) == 24:
            del self.ang4_tracker[0]
        if len(self.headpoint_tracker) == 24:
            del self.headpoint_tracker[0]

        if is_pushup:
            return "Pushup"
        elif is_plank:
            return "Plank"
        elif is_squat:
            return "Squat"

        return None

    def estimate(self) -> None:
        """ Estimate pose (base function) """
        if self.video_reader.is_opened() is False:
            print("Error File Not Found.")

        out = cv2.VideoWriter("output.avi", self.fourcc, self.video_fps, (self.width, self.height))
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
    """ Sub: Pushup class """
    def __init__(self, video_reader) -> None:
        super().__init__(video_reader)
        self.video_reader = video_reader
        self.pushups_count = 0
        self.is_pushup = False

    def _draw(self, image):
        """ Draw lines between shoulder, wrist and foot """
        left_shoulder_wrist_foot = self.is_point_in_keypoints("left_shoulder") and \
                                    self.is_point_in_keypoints("left_wrist") and \
                                    self.is_point_in_keypoints("left_foot_index")
        right_shoulder_wrist_foot = self.is_point_in_keypoints("right_shoulder") and \
                                    self.is_point_in_keypoints("right_wrist") and \
                                    self.is_point_in_keypoints("right_foot_index")
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
        """ Pushup algorithm """
        # Distance algorithm
        head_point = self.get_available_point(["nose", "left_ear", "right_ear", "left_eye", "right_eye"])
        ankle = self.get_available_point(["left_ankle", "right_ankle"])
        if head_point is None or ankle is None:
            return

        diff_y = self.operation.dist_y(head_point, ankle)

        # Angle algorithm
        head_pos = self.operation.point_position(head_point, (self.width / 2, 0), (self.width / 2, self.height))
        wrist = self.get_available_point(["left_wrist", "right_wrist"])
        ang = self.operation.angle(head_point, ankle, wrist)
        if diff_y < 250 and (ang < 40 and head_pos == "right") or (ang > 140 and head_pos == "left"):
            self.is_pushup = True
        if diff_y > 300 and self.is_pushup is True:
            self.pushups_count += 1
            self.is_pushup = False


    def measure(self) -> None:
        """ Measure pushups (base function) """
        if self.video_reader.is_opened() is False:
            print("Error File Not Found.")

        out = cv2.VideoWriter("output.avi", self.fourcc, self.video_fps, (self.width, self.height))
        pushup_count_prev = pushup_count_current = progress_counter = 0
        progress_bar_color = (255, 255, 255)
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

            # overlay
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = self.draw.overlay(image)

            # progress bar
            image = cv2.rectangle(image, (0, self.height//8 - 10), (self.width//10 * progress_counter, self.height//8),
                                        progress_bar_color, cv2.FILLED)
            if results.pose_landmarks is not None:
                self.key_points = self.get_keypoints(image, results)
                self.pose_algorithm()
                image = self._draw(image)
                image = self.draw.pose_text(image, "Pushups Count: " + str(self.pushups_count))
                pushup_count_prev = pushup_count_current
                pushup_count_current = self.pushups_count
                if self.pushups_count > 0 and abs(pushup_count_current - pushup_count_prev) == 1:
                    progress_counter += 1
                    if progress_counter == 10:
                        progress_counter = 0
                        progress_bar_color = random.choices(range(128, 256), k=3)

            out.write(image)
            cv2.imshow('Pushups', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.video_reader.release()


class Plank(Pose):
    """ Sub: Plank class """
    def __init__(self, video_reader) -> None:
        super().__init__(video_reader)
        self.video_reader = video_reader
        self.timer = Timer()
        self.plank_counter = 0
        self.start_time = None
        self.total_time = 0

    def _draw(self, image):
        """ Draw lines between shoulder, wrist and foot """
        left_shoulder_wrist_foot = self.is_point_in_keypoints("left_shoulder") and \
                                    self.is_point_in_keypoints("left_elbow") and \
                                    self.is_point_in_keypoints("left_foot_index")
        right_shoulder_wrist_foot = self.is_point_in_keypoints("right_shoulder") and \
                                    self.is_point_in_keypoints("right_elbow") and \
                                    self.is_point_in_keypoints("right_foot_index")
        if left_shoulder_wrist_foot:
            image = self.draw.draw_line(image, self.get_point("left_shoulder"), self.get_point("left_elbow"))
            image = self.draw.draw_line(image, self.get_point("left_elbow"), self.get_point("left_foot_index"))
            image = self.draw.draw_line(image, self.get_point("left_foot_index"), self.get_point("left_shoulder"))
        elif right_shoulder_wrist_foot:
            image = self.draw.draw_line(image, self.get_point("right_shoulder"), self.get_point("right_elbow"))
            image = self.draw.draw_line(image, self.get_point("right_elbow"), self.get_point("right_foot_index"))
            image = self.draw.draw_line(image, self.get_point("right_foot_index"), self.get_point("right_shoulder"))
        else:
            pass
        return image

    def pose_algorithm(self):
        """ Plank algorithm """
        ang1 = ang2 = ang3 = ang4 = None
        is_plank = False

        # Calculate angle between lines shoulder-elbow, elbow-wrist
        if self.is_point_in_keypoints("left_shoulder") and \
            self.is_point_in_keypoints("left_elbow") and \
            self.is_point_in_keypoints("left_wrist"):
            ang1 = self.two_line_angle("left_shoulder", "left_elbow", "left_wrist")
        elif self.is_point_in_keypoints("right_shoulder") and \
            self.is_point_in_keypoints("right_elbow") and \
            self.is_point_in_keypoints("right_wrist"):
            ang1 = self.two_line_angle("right_shoulder", "right_elbow", "right_wrist")
        else:
            pass

        # Calculate angle between lines shoulder-hip, hip-ankle
        if self.is_point_in_keypoints("left_shoulder") and \
            self.is_point_in_keypoints("left_hip") and \
            self.is_point_in_keypoints("left_ankle"):
            ang2 = self.two_line_angle("left_shoulder", "left_hip", "left_ankle")
        elif self.is_point_in_keypoints("right_shoulder") and \
            self.is_point_in_keypoints("right_hip") and \
            self.is_point_in_keypoints("right_ankle"):
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
                    self.plank_counter += 1
                    self.ang1_tracker.append(ang1)
                    self.ang4_tracker.append(ang4)

        if self.plank_counter >= 24 and len(self.ang1_tracker) == 24 and len(self.ang4_tracker) == 24:
            ang1_diff1 = abs(self.ang1_tracker[0] - self.ang1_tracker[12])
            ang1_diff2 = abs(self.ang1_tracker[12] - self.ang1_tracker[23])
            ang1_diff_mean = (ang1_diff1 + ang1_diff2) / 2
            ang4_mean = sum(self.ang4_tracker) / len(self.ang4_tracker)
            del self.ang1_tracker[0]
            del self.ang4_tracker[0]
            if ang1_diff_mean < 5 and not 75 <= ang4_mean <= 105:
                is_plank = True
                if self.start_time is None:
                    self.timer.start()
                self.start_time = self.timer._start_time
            else:
                is_plank = False
                if self.start_time is not None:
                    self.timer.end()
                    self.total_time = self.timer._total_time
                self.start_time = None

    def measure(self) -> None:
        """ Measure planks (base function) """
        if self.video_reader.is_opened() is False:
            print("Error File Not Found.")

        time_adjustment = 6 / self.video_fps # 6 is magic number
        progress_counter = 0
        progress_bar_color = (255, 255, 255)
        out = cv2.VideoWriter("output.avi", self.fourcc, self.video_fps, (self.width, self.height))
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

            # overlay
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = self.draw.overlay(image)

            # progress bar
            image = cv2.rectangle(image, (0, self.height//8 - 10), (self.width//60 * progress_counter, self.height//8),
                                        progress_bar_color, cv2.FILLED)
            if results.pose_landmarks is not None:
                self.key_points = self.get_keypoints(image, results)
                self.pose_algorithm()
                image = self._draw(image)
                time = round(time_adjustment * (self.timer.get_current_time() + self.total_time), 2)
                h_m_s_ms_time = self.timer.convert_time(time) # convert Seconds to Hour : Minute : Second : Milli-Second format
                image = self.draw.pose_text(image, "Plank Timer: " + str(h_m_s_ms_time)[:10])
                progress_counter = int(int(time) % self.video_fps)

            out.write(image)
            cv2.imshow('Planks', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.video_reader.release()


class Squat(Pose):
    """ Sub: Squat class """
    def __init__(self, video_reader) -> None:
        super().__init__(video_reader)
        self.video_reader = video_reader
        self.squats_count = 0
        self.is_squat = False

    def pose_algorithm(self):
        """ Squat algorithm """
        # Distance algorithm
        head_point = self.get_available_point(["nose", "left_ear", "right_ear", "left_eye", "right_eye"])
        ankle = self.get_available_point(["left_ankle", "right_ankle"])
        if head_point is None or ankle is None:
            return

        diff_y = self.operation.dist_y(head_point, ankle)
        norm_diff_y = self.operation.normalize(diff_y, 0, self.height)
        if norm_diff_y < 0.5:
            self.is_squat = True
        if norm_diff_y > 0.5 and self.is_squat is True:
            self.squats_count += 1
            self.is_squat = False

    def measure(self) -> None:
        """ Measure squats (base function) """
        if self.video_reader.is_opened() is False:
            print("Error File Not Found.")

        out = cv2.VideoWriter("output.avi", self.fourcc, self.video_fps, (self.width, self.height))
        squads_count_prev = squad_count_current = progress_counter = 0
        progress_bar_color = (255, 255, 255)
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

            # overlay
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = self.draw.overlay(image)
            image = self.draw.skeleton(image, results)

            # progress bar
            image = cv2.rectangle(image, (0, self.height//8 - 10), (self.width//10 * progress_counter, self.height//8),
                                        progress_bar_color, cv2.FILLED)
            if results.pose_landmarks is not None:
                self.key_points = self.get_keypoints(image, results)
                self.pose_algorithm()
                image = self.draw.pose_text(image, "Squats Count: " + str(self.squats_count))
                squads_count_prev = squad_count_current
                squad_count_current = self.squats_count
                if self.squats_count > 0 and abs(squad_count_current - squads_count_prev) == 1:
                    progress_counter += 1
                    if progress_counter == 10:
                        progress_counter = 0
                        progress_bar_color = random.choices(range(128, 256), k=3)

            out.write(image)
            cv2.imshow('Squats', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.video_reader.release()
