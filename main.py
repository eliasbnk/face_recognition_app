import cv2 as cv
import logging
from face_identification import FaceIdentification
from encoding_provider import EncodingProvider

logging.basicConfig(filename='video_stream.log', level=logging.DEBUG,
                    format='%(asctime)s - %(message)s')


class VideoStream:
    def __init__(self, camera_id=0,
                 facial_encodings_file=None, known_faces_dir=None ):
        self.camera_id = camera_id
        self.MIRROR_VIDEO = True
        self.WINDOW_TITLE = 'Live Video Feed'
        self.WINDOW_SIZE = (1024, 720)
        self.EXIT_KEY = 'q'
        self.face_identification = None
        self.facial_encodings_file = facial_encodings_file
        self.known_faces_dir = known_faces_dir
        self.encoding_provider = EncodingProvider(facial_encodings_file=self.facial_encodings_file,
                                                  known_faces_dir=self.known_faces_dir, mirror_video=self.MIRROR_VIDEO)
        
        if facial_encodings_file is not None or known_faces_dir is not None:
            self.facial_encodings_file = self.encoding_provider.get_encoding_data()
            if self.facial_encodings_file is not None:
                if self.WINDOW_TITLE == 'Live Video Feed':
                    self.WINDOW_TITLE = 'Live Face Recognition'
                self.face_identification = FaceIdentification(self.facial_encodings_file)

        self.process_this_frame = True

        try:
            self.video_stream = self._initialize_video_stream()
        except Exception as e:
            logging.error(f"Failed to initialize video stream: {e}")

    def _initialize_video_stream(self):
        try:
            stream = cv.VideoCapture(self.camera_id)
            if not stream.isOpened():
                logging.error("Unable to open the camera. Please check the connection.")
                return None
            return stream
        except Exception as e:
            logging.error(f"Error initializing video stream: {e}")
            return None

    def _capture_frame(self):
        try:
            frame_captured, frame = self.video_stream.read()
            if not frame_captured:
                logging.warning("Frame capture failed. The video stream may be interrupted.")
                return None
            return cv.flip(frame, 1) if self.MIRROR_VIDEO else frame
        except Exception as e:
            logging.error(f"Error capturing frame: {e}")
            return None

    def _display_frame(self, frame):
        try:
            # Resize the frame to the specified window size
            resized_frame = cv.resize(frame, self.WINDOW_SIZE)
            cv.imshow(self.WINDOW_TITLE, resized_frame)
        except Exception as e:
            logging.error(f"Error displaying frame: {e}")

    def _is_exit_key_pressed(self):
        try:
            return cv.waitKey(1) == ord(self.EXIT_KEY)
        except Exception as e:
            logging.error(f"Error checking exit key: {e}")
            return False

    def _stop_video_stream(self):
        try:
            if self.video_stream:
                self.video_stream.release()
        except Exception as e:
            logging.error(f"Error stopping video stream: {e}")

    @staticmethod
    def _close_window():
        try:
            cv.destroyAllWindows()
        except Exception as e:
            logging.error(f"Error closing window: {e}")

    def _run_video_stream_loop(self):
        while True:
            frame = self._capture_frame()
            if frame is None:
                break

            if self.face_identification:
                processed_frame = self.face_identification.process_frame(frame)
            else:
                processed_frame = frame

            self._display_frame(processed_frame)

            if self._is_exit_key_pressed():
                break

        self._stop_video_stream()
        self._close_window()

    def start_streaming(self):
        if self.video_stream is not None:
            self._run_video_stream_loop()
        else:
            logging.error("Video stream is not initialized. Exiting.")


if __name__ == "__main__":
    video_stream = VideoStream(facial_encodings_file="facial_encodings.pkl")
    video_stream.start_streaming()
