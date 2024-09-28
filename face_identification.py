import pickle
import cv2 as cv
import numpy as np
import face_recognition
import logging
from attendance_logger import AttendanceLogger

logging.basicConfig(filename='face_identification.log', level=logging.DEBUG,
                    format='%(asctime)s - %(message)s')


class FaceIdentification:
    def __init__(self, facial_encodings_file):
        self.known_face_encodings, self.known_face_names = self._load_known_people(facial_encodings_file)
        self.MIN_CONFIDENCE_THRESHOLD = 0.60
        self.attendee_name = None
        self.attendance_logger = AttendanceLogger()

    def _load_known_people(self, facial_encodings_file):
        encoded_faces = self._load_data_from_file(facial_encodings_file)
        if encoded_faces is None:
            logging.warning("No encoded faces found.")
            return [], []

        return self._extract_face_encodings_and_names(encoded_faces)

    @staticmethod
    def _load_data_from_file(facial_encodings_file):
        try:
            with open(facial_encodings_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading {facial_encodings_file}: {e}")
            return None

    @staticmethod
    def _extract_face_encodings_and_names(encoded_faces):
        known_face_encodings = []
        known_face_names = []

        for key, value in encoded_faces.items():
            known_face_names.append(key)
            known_face_encodings.extend(value)

        return known_face_encodings, known_face_names

    def process_frame(self, frame):
        small_frame = self._resize_frame(frame)
        rgb_small_frame = self._convert_color_to_rgb(small_frame)

        face_locations, face_encodings = self._detect_faces(rgb_small_frame)

        face_names = self._identify_faces(face_encodings)

        return self._draw_boxes_around_faces(frame, face_locations, face_names)

    @staticmethod
    def _resize_frame(frame):
        return cv.resize(frame, (0, 0), fx=0.25, fy=0.25)

    @staticmethod
    def _convert_color_to_rgb(small_frame):
        return np.ascontiguousarray(small_frame[:, :, ::-1])

    @staticmethod
    def _detect_faces(rgb_small_frame):
        try:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            return face_locations, face_encodings
        except Exception as e:
            logging.error(f"Error detecting faces: {e}")
            return [], []

    def _identify_faces(self, face_encodings):
        face_data = []
        for face_encoding in face_encodings:
            name, confidence = self._get_face_name(face_encoding)
            if name is not None:
                face_data.append((name, confidence))
        return face_data

    def _get_face_name(self, face_encoding):
        """
        Gets the name associated with a given face encoding and returns the confidence score.

        Parameters:
        - face_encoding: The encoding of the face to identify.

        Returns:
        - A tuple containing the name associated with the face and the confidence score,
          or ("Unknown", 0.0) if no match is found.
        """
        if len(self.known_face_encodings) == 0:
            logging.warning("No known face encodings available.")
            return "Unknown", 0.0

        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

        if len(face_distances) == 0:
            logging.warning("No face distances computed, returning 'Unknown'.")
            return "Unknown", 0.0

        best_match_index = np.argmin(face_distances)
        confidence_score = 1 - face_distances[best_match_index]

        if confidence_score < self.MIN_CONFIDENCE_THRESHOLD:
            return None, None

        if best_match_index < len(self.known_face_encodings):
            matched_name = self.known_face_names[best_match_index]
            name_without_id = matched_name.split('_')[0]
            self.attendee_name = name_without_id

            self.attendance_logger.log_attendance(self.attendee_name)
            return name_without_id, confidence_score

        return "Unknown", 0.0

    def _draw_boxes_around_faces(self, frame, face_locations, face_data):
        for (top, right, bottom, left), (name, confidence) in zip(face_locations, face_data):
            self._draw_box_around_face(frame, top, right, bottom, left, name, confidence)
        return frame

    @staticmethod
    def _draw_box_around_face(frame, top, right, bottom, left, name, confidence):
        try:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv.rectangle(frame, (left, top), (right, bottom), color, 2)

            font = cv.FONT_HERSHEY_COMPLEX
            cv.putText(frame, name, (left, top - 10), font, 1.0, color, 2)

            confidence_text = f"{confidence * 100:.2f}%"
            cv.putText(frame, confidence_text, (right - 150, top - 10), font, 1.0, color, 2)

        except Exception as e:
            logging.error(f"Error drawing box around face: {e}")

