import cv2

class VideoReader:
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame = 0

    def read_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret == False or frame is None:
                return None
            self._current_frame += 1
        else:
            return None
        return frame

    def read_n_frames(self, n=1):
        frames_list = []
        for i in range(n):
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret == False or frame is None:
                    return None
                frames_list.append(frame)
                self._current_frame += 1
            else:
                return None
        return frames_list

    def is_opened(self):
        return self.cap.isOpened()

    def get_frame_width(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def get_frame_height(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_video_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_current_frame(self):
        return self._current_frame

    def get_total_frames(self):
        return self._total_frames

    def release(self):
        self.cap.release()

    def __del__(self):
        self.release()