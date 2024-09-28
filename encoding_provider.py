import os
import pickle
import numpy as np
from image_encoder import ImageEncoder
import logging

# Configure logging
logging.basicConfig(filename='encoding_provider.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class EncodingProvider:
    def __init__(self, facial_encodings_file, known_faces_dir, valid_extensions=None, mirror_video=True):
        self.mirror_video=mirror_video
        self.valid_extensions = valid_extensions or {'.jpg', '.jpeg', '.png'}
        self.facial_encodings_file = facial_encodings_file
        self.known_faces_dir = known_faces_dir
        self.encoder = ImageEncoder(known_faces_dir=self.known_faces_dir, mirror_video=self.mirror_video)

    @staticmethod
    def _is_valid_file(path):
        """Check if the provided file exists and is not empty."""
        try:
            valid = os.path.isfile(path) and os.path.getsize(path) > 0
            if valid:
                logging.info(f"Valid file found: {path}")
            else:
                logging.warning(f"File '{path}' is invalid or empty.")
            return valid
        except Exception as e:
            logging.error(f"Error checking file validity for '{path}': {e}")
            return False

    def _is_valid_encoding_file(self):
        """Check if the encoding file holds a valid mapping of image encodings to person names."""
        if not self._is_valid_file(self.facial_encodings_file):
            logging.error(f"Encoding file '{self.facial_encodings_file}' is invalid.")
            return False

        try:
            with open(self.facial_encodings_file, 'rb') as file:
                contents = pickle.load(file)
            if not isinstance(contents, dict):
                logging.error(f"Encoding file '{self.facial_encodings_file}' does not contain a valid dictionary.")
                return False

            for name, encodings in contents.items():
                if not isinstance(name, str) or not isinstance(encodings, list):
                    logging.error(f"Invalid entry for name '{name}': must be a string with a list of encodings.")
                    return False
                if not all(isinstance(enc, np.ndarray) for enc in encodings):
                    logging.error(f"All encodings for '{name}' must be numpy arrays.")
                    return False
                if not encodings:
                    logging.error(f"No encodings found for '{name}'.")
                    return False

            logging.info(f"Encoding file '{self.facial_encodings_file}' validated successfully.")
            return True
        except (pickle.UnpicklingError, FileNotFoundError) as e:
            logging.error(f"Error loading encoding file '{self.facial_encodings_file}': {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error in encoding file validation for '{self.facial_encodings_file}': {e}")
            return False

    def _is_valid_faces_dir(self):
        """Check if the provided folder exists and contains valid image files."""
        if not os.path.isdir(self.known_faces_dir):
            logging.error(f"'{self.known_faces_dir}' is not a valid directory.")
            return False

        try:
            valid_images_found = False
            for entry in os.listdir(self.known_faces_dir):
                child_path = os.path.join(self.known_faces_dir, entry)

                # Ignore non-directory entries
                if os.path.isdir(child_path):
                    for file in os.listdir(child_path):
                        file_path = os.path.join(child_path, file)
                        # Check for valid image files
                        if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in self.valid_extensions:
                            valid_images_found = True
                            logging.info(f"Valid image file found: {file_path}")
                        else:
                            logging.warning(f"'{file_path}' is not a valid image file or has an unsupported extension.")
                else:
                    logging.warning(f"'{child_path}' is not a directory or it is a file.")

            if not valid_images_found:
                logging.warning(f"No valid image files found in directory '{self.known_faces_dir}'.")
            return valid_images_found
        except Exception as e:
            logging.error(f"Error validating faces directory '{self.known_faces_dir}': {e}")
            return False

    def get_encoding_data(self):
        try:
            # Both facial_encodings_file and known_faces_dir are provided
            if self.facial_encodings_file and self.known_faces_dir:
                if not self._is_valid_encoding_file() and not self._is_valid_faces_dir():
                    logging.error("Both the encoding file and the images directory are invalid or empty.")
                    return None

                # Both are valid
                if self._is_valid_encoding_file() and self._is_valid_faces_dir():
                    logging.info("Both encoding file and images directory are valid.")
                    return self.facial_encodings_file

                # Encoding file is valid but folder is invalid
                if self._is_valid_encoding_file() and not self._is_valid_faces_dir():
                    logging.error("The images directory is invalid. Using the valid encoding file only.")
                    return self.facial_encodings_file

                # Folder is valid but encoding file is invalid
                if self._is_valid_faces_dir() and not self._is_valid_encoding_file():
                    logging.error("The encoding file is invalid. Building new encodings from the valid images "
                                  "directory.")
                    return self.encoder.process_images()
            # Provided facial_encodings_file only
            elif self.facial_encodings_file and self._is_valid_encoding_file():
                logging.info("Using the valid encoding file.")
                return self.facial_encodings_file
            # Provided known_faces_dir only
            elif self.known_faces_dir and self._is_valid_faces_dir():
                logging.info("Building encoding file from the valid images directory.")
                self.facial_encodings_file = self.encoder.process_images()
                return self.facial_encodings_file

            # If no valid parameters were provided, log an error
            logging.error("No valid encoding file or images directory provided. Unable to obtain encoding data.")
            return None

        except Exception as e:
            logging.error(f"An unexpected error occurred while getting encoding data: {e}")
            return None
