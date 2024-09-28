import os  # To work with directories and files
import pickle  # To save data in byte stream
import face_recognition  # To recognize faces in pictures
import logging  # To track what happens while the program runs
import json  # To work with data in JSON format
import cv2 as cv  # To process images and frames

# Configure logging
logging.basicConfig(filename='image_encoder.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ImageEncoder:
    """
    This code assumes your folder structure is as follows:

    image_db/
        student_id_1/
            image1.jpg
            image2.jpg
            ...
        student_id_2/
            ...

    Only .jpeg, .jpg, and .png image files are processed by default.
    If you want to process more or fewer file types, you can pass them into the `valid_extensions` parameter when setting up the class.

    The program also keeps track of images it has already processed by saving their file paths in 'processed_images.json'.
    If an image has been processed before, it will be skipped, so we don't waste time and resources encoding it again.
    """

    def __init__(self, known_faces_dir='image_db', processed_file='processed_images.json',
                 facial_encodings_file='facial_encodings.pkl',
                 cropped_faces_dir='cropped_faces', valid_extensions=None, mirror_video=True):
        self.mirror_video = mirror_video
        self.known_faces_dir = known_faces_dir  # Set the main folder for images
        self.processed_file = processed_file  # File to track processed images
        self.facial_encodings_file = facial_encodings_file  # File for saving facial encodings
        self.cropped_faces_dir = cropped_faces_dir  # Directory to save cropped faces
        self.CROPPED_IMAGE_SIZE = (50, 50)  # Pixel size for cropped faces
        self.student_data = {}  # Hashmap to hold student IDs and their encodings

        # Set valid image extensions
        self.valid_extensions = valid_extensions or {'.jpg', '.jpeg', '.png'}

        # Create the cropped faces directory if it doesn't exist
        self._create_cropped_faces_directory()
        self._initialize_encoding_file()  # Set up the output file

    def _initialize_encoding_file(self):
        """Creates a new empty file for facial encodings if it does not exist."""
        try:
            if not os.path.exists(self.facial_encodings_file):  # Check if the output file does not exist
                logging.info(f"{self.facial_encodings_file} does not exist. Creating a new file.")  # Log the creation
                with open(self.facial_encodings_file, 'wb') as f:  # Open a new file to write
                    pickle.dump({}, f)  # Save an empty hashmap in the new file
        except Exception as e:  # Handle file creation errors
            logging.error(f"Error creating the output file: {str(e)}")  # Log the error

    def _create_cropped_faces_directory(self):
        """Creates the cropped faces directory if it doesn't exist."""
        try:
            if not os.path.exists(self.cropped_faces_dir):  # Check if the directory exists
                os.makedirs(self.cropped_faces_dir)  # Create directory
                logging.info(f"Created directory: {self.cropped_faces_dir}")  # Log creation
        except Exception as e:  # Handle errors during directory creation
            logging.error(f"Error creating directory {self.cropped_faces_dir}: {str(e)}")  # Log error

    def process_images(self):
        try:

            processed_images = self._load_processed_images()

            for student_dir in self._get_student_directories(self.known_faces_dir):
                student_id = os.path.basename(student_dir)  # Use the child directory name as student ID
                for img_path in self._get_image_paths_from_directory(student_dir):
                    if img_path in processed_images:  # Skip already processed images
                        logging.info(f"Skipping already processed image: {img_path}")
                        continue

                    encoding, cropped_face = self._encode_image(img_path)  # Get encoding and cropped face

                    if encoding is None or cropped_face is None:
                        logging.warning(f"Encoding failed for image: {img_path}. Skipping.")
                        continue

                    unique_student_id = self._generate_unique_student_id(student_id)
                    self.student_data[unique_student_id] = [encoding]
                    self._save_cropped_image(cropped_face, img_path)  # Save the cropped face
                    self._save_processed_image(img_path)  # Mark the image as processed
                    self._remove_image(img_path)  # Remove the original image

            self._save_encodings()  # Save all encodings to the file
            return self.facial_encodings_file
        except Exception as e:  # Handle errors during the build process
            logging.error(f"Error during image processing: {str(e)}")  # Log the error

    def _generate_unique_student_id(self, base_id):
        """
        Generates a unique student ID by appending a number.

        Parameters:
        - base_id: The base student ID.

        Returns:
        - A unique student ID with a number suffix or None if an error occurs.
        """
        if not base_id:
            logging.warning("Base student ID is empty. Cannot generate unique ID.")
            return None

        try:
            count = 1
            unique_id = f"{base_id}_{count}"

            # Check if this unique ID already exists
            while unique_id in self.student_data:
                count += 1  # Increment count if ID exists
                unique_id = f"{base_id}_{count}"  # Generate new unique ID

            logging.info(f"Generated unique student ID: {unique_id}.")  # Log the unique ID
            return unique_id  # Return the unique ID
        except Exception as e:  # Handle errors during unique ID generation
            logging.error(f"Error generating unique student ID from {base_id}: {str(e)}")  # Log the error
            return None  # Return None on error

    @staticmethod
    def _get_student_directories(directory):
        """Retrieves all student directories from the specified directory."""
        student_dirs = []
        try:
            for entry in os.scandir(directory):  # Scan the main directory for student directories
                if entry.is_dir() and not entry.name.startswith(
                        '.'):  # Skip bad directories or hidden ones
                    student_dirs.append(entry.path)  # Add valid directory path to the list
            logging.info(
                f"Found {len(student_dirs)} valid student directory(s) in {directory}.")  # Log found directories
        except Exception as e:  # Handle directory access errors
            logging.error(f"Error accessing directory {directory}: {str(e)}")  # Log the error
        return student_dirs

    def _get_image_paths_from_directory(self, directory):
        """
        Retrieves paths of images from the specified directory.

        Parameters:
        - directory: The directory to search for images.

        Returns:
        - A list of image paths.
        """
        image_paths = []
        try:
            for root, _, files in os.walk(directory):  # Walk through the directory
                for file in files:
                    if file.lower().endswith(tuple(self.valid_extensions)):  # Check for valid image file types
                        image_paths.append(os.path.join(root, file))  # Add image path to the list
            logging.info(f"Found {len(image_paths)} image(s) in {directory}.")  # Log found images
        except Exception as e:  # Handle directory access errors
            logging.error(f"Error accessing directory {directory}: {str(e)}")  # Log the error
        return image_paths

    def _encode_image(self, img_path):
        """
        Encodes the image and returns its encoding and the cropped face.

        Parameters:
        - img_path: The path of the image to encode.

        Returns:
        - A tuple containing the encoding and the cropped face image or (None, None) if an error occurs.
        """
        try:

            image = face_recognition.load_image_file(img_path)
            if self.mirror_video:
                image = cv.flip(image, 1)
            encodings = face_recognition.face_encodings(image)
            if not encodings:  # Guard clause for no face found
                logging.warning(f"No faces found in image: {img_path}.")  # Log warning
                return None, None  # Return None if no encoding found

            encoding = encodings[0]  # Get the first encoding
            top, right, bottom, left = face_recognition.face_locations(image)[0]  # Get the location of the first face

            cropped_face = image[top:bottom, left:right]
            # Resize cropped face to desired size
            cropped_face = cv.resize(cropped_face, self.CROPPED_IMAGE_SIZE)

            logging.info(f"Encoded image: {img_path} successfully.")  # Log successful encoding
            return encoding, cropped_face  # Return encoding and cropped face
        except Exception as e:  # Handle errors during encoding
            logging.error(f"Error encoding image {img_path}: {str(e)}")  # Log the error
            return None, None  # Return None if an error occurs

    def _save_cropped_image(self, cropped_face, img_path):
        """
        Saves the cropped face image to the specified directory.

        Parameters:
        - cropped_face: The cropped face image.
        - img_path: The original image path for naming.
        """
        try:
            base_name = os.path.splitext(os.path.basename(img_path))[0]  # Get base name without extension
            save_path = os.path.join(self.cropped_faces_dir, f"{base_name}_cropped.jpg")  # Path for cropped image
            cv.imwrite(save_path, cropped_face)  # Save the cropped image
            logging.info(f"Saved cropped image to {save_path}.")  # Log successful save
        except Exception as e:  # Handle errors during saving
            logging.error(f"Error saving cropped image {img_path}: {str(e)}")  # Log the error

    def _load_processed_images(self):
        """Loads the list of processed images from the processed_file."""
        try:
            if os.path.exists(self.processed_file):  # Check if the processed file exists
                with open(self.processed_file, 'r') as f:  # Open the processed file for reading
                    return json.load(f)  # Load the JSON data
            return []  # Return empty list if no file found
        except Exception as e:  # Handle loading errors
            logging.error(f"Error loading processed images: {str(e)}")  # Log the error
            return []  # Return empty list on error

    def _save_processed_image(self, img_path):
        """Saves a processed image to the processed_file."""
        try:
            processed_images = self._load_processed_images()  # Load existing processed images
            processed_images.append(img_path)  # Add the new processed image
            with open(self.processed_file, 'w') as f:  # Open the processed file for writing
                json.dump(processed_images, f)  # Save updated list
            logging.info(f"Marked image as processed: {img_path}")  # Log saving
        except Exception as e:  # Handle saving errors
            logging.error(f"Error saving processed image {img_path}: {str(e)}")  # Log the error

    @staticmethod
    def _remove_image(img_path):
        """Removes the original image from disk."""
        try:
            os.remove(img_path)  # Remove the image
            logging.info(f"Removed image: {img_path}")  # Log removal
        except Exception as e:  # Handle removal errors
            logging.error(f"Error removing image {img_path}: {str(e)}")  # Log the error

    def _save_encodings(self):
        """Saves the facial encodings to a specified file."""
        try:
            with open(self.facial_encodings_file, 'wb') as f:  # Open the encoding file for writing
                pickle.dump(self.student_data, f)  # Save the student data to the file
            logging.info(f"Saved facial encodings to {self.facial_encodings_file}.")  # Log successful save
        except Exception as e:  # Handle saving errors
            logging.error(f"Error saving facial encodings: {str(e)}")  # Log the error


if __name__ == "__main__":
    encoder = ImageEncoder()  # Create an instance of the encoder
    encoder.process_images()  # Process the images
