import os  # To work with directories and files
import pickle  # To save data in byte stream
import face_recognition  # To recognize faces in pictures
import logging  # To track what happens while the program runs
import json  # To work with data in JSON format
import cv2  # To process images and frames


class ImageEncoder:
    """
        This code assumes your folder structure is as follows:

        image_db/
            lec_num_1/
                student_id_1/
                    image1.jpg
                    image2.jpg
                    ...
                student_id_2/
                    ...
            lec_num_2/
                ...

        Only .jpeg, .jpg, and .png image files are processed by default.
        If you want to process more or fewer file types, you can pass them into the `image_types` parameter when setting up the class.

        The program also keeps track of images it has already processed by saving their file paths in 'processed_images.json'.
        If an image has been processed before, it will be skipped, so we don't waste time and resources encoding it again.
        """

    def __init__(self, parent_dir='image_db', output_file='facial_data.pkl', processed_file='processed_images.json',
                 cropped_faces_dir='cropped_faces', image_types=None):
        """
        Initialize the ImageEncoder class.

        Parameters:
        - parent_dir: The main folder where images are stored. Default is 'image_db'.
        - output_file: Name of the file where face data will be saved. Default is 'facial_data.pkl'.
        - processed_file: Name of the JSON file that keeps track of processed images. Default is 'processed_images.json'.
        - cropped_faces_dir: Folder where cropped images will be saved. Default is 'cropped_faces'.
        """
        self.parent_dir = parent_dir  # Set the main folder where images are stored
        self.output_file = output_file  # Set the file name to save face data
        self.processed_file = processed_file  # Set the JSON file to keep track of processed images
        self.cropped_faces_dir = cropped_faces_dir  # Set the folder for cropped images
        self.student_data = {}  # This will hold student IDs and their face data
        self.CROPPED_IMAGE_SIZE = (50, 50)  # Size for cropped images in pixels
        self.image_types = image_types if image_types else ['jpeg', 'jpg', 'png']  # The image types we will process
        self._initialize_pickle_file()  # Set up the output file
        self._initialize_cropped_faces_dir()  # Set up the cropped faces folder

    def run(self):
        """
        Start the process to encode images.

        This function checks for images and encodes them if they haven't been processed yet.
        """
        try:
            self._process_and_encode_images()  # Call the function to process and encode images
            self._save_encodings()  # Call the function to save encoded data
        except Exception as e:  # If there is an error during the encoding process
            logging.error(f"Error during image encoding: {str(e)}")  # Log the error

    def _initialize_pickle_file(self):
        """
        Check if the output file exists. If not, creates a new empty file to store face data.
        """
        if not os.path.exists(self.output_file):  # Check if the output file does not exist
            logging.info(
                f"{self.output_file} does not exist. Creating a new file.")  # Log that the file is being created
            try:
                with open(self.output_file, 'wb') as f:  # Open a new file to write data
                    pickle.dump({}, f)  # Save an empty hashmap in the new file
            except Exception as e:  # If there is an error during file creation
                logging.error(f"Error creating the output file: {str(e)}")  # Log the error

    def _initialize_cropped_faces_dir(self):
        """
        Check if the folder for cropped images exists. If not, creates the folder.
        """
        try:
            os.makedirs(self.cropped_faces_dir, exist_ok=True)  # Create the folder if it doesn't exist
            logging.info(f"Cropped faces directory is set to {self.cropped_faces_dir}")  # Log the setup of the folder
        except Exception as e:  # If there is an error creating the folder
            logging.error(f"Error creating cropped faces directory: {str(e)}")  # Log the error

    def _process_and_encode_images(self):
        """
        Go through each lecture folder and process student's images.
        """
        try:
            lecture_numbers = self._get_lecture_numbers()  # Get a list of lecture folders
            if not lecture_numbers:  # Check if there are no lecture folders
                logging.warning(f"No lecture directories found in {self.parent_dir}. Exiting process.")  # Log the error
                return  # Exit the function if no lectures found

            for lecture_number in lecture_numbers:  # Loop through each lecture folder
                lecture_path = os.path.join(self.parent_dir, lecture_number)  # Build the path for the lecture folder
                if os.path.isdir(lecture_path):  # Check if the path is indeed a folder
                    students = os.listdir(lecture_path)  # Get the list of student folders
                    if not students:  # Check if there are no student folders
                        logging.warning(
                            f"No student directories found in {lecture_path}. Skipping this lecture.")  # Log the error
                        continue  # Skip to the next lecture

                    self._process_students_in_lecture(lecture_path, students)  # Process students in this lecture
        except Exception as e:  # If there is an error during processing
            logging.error(f"Error processing images: {str(e)}")  # Log the error

    def _get_lecture_numbers(self):
        """
        Get a list of lecture folder names found in the parent directory.
        """
        if not os.path.exists(self.parent_dir):  # Check if the main directory does not exist
            logging.error(f"Parent directory {self.parent_dir} does not exist.")  # Log error
            return []  # Return an empty list if the directory is missing

        try:
            lectures = os.listdir(self.parent_dir)  # Get the list of all folders in parent directory
            if not lectures:  # Check if there are no lecture folders
                logging.warning(f"No lecture directories found in {self.parent_dir}.")  # Log the error
            return lectures  # Return the list of lecture folders
        except Exception as e:  # If there is an error during listing
            logging.error(f"Error getting lecture numbers: {str(e)}")  # Log the error
            return []  # Return an empty list on error

    def _process_students_in_lecture(self, lecture_path, students):
        """
        Go through each student folder and encode their images.

        Parameters:
        - lecture_path: Path to the lecture folder.
        - students: List of student directories.
        """
        try:
            for student_id in students:  # Loop through each student folder
                student_path = os.path.join(lecture_path, student_id)  # Build the path for the student
                if os.path.isdir(student_path):  # Check if the path is a folder
                    image_files = os.listdir(student_path)  # Get list of image files for this student
                    if not image_files:  # Check if no images are found
                        logging.warning(
                            f"No images found for student {student_id} in {student_path}. Skipping student.")  # Log the error
                        continue  # Skip to the next student

                    self._encode_student_images(student_id,
                                                student_path)  # Call function to encode images for this student
        except Exception as e:  # If there is an error during processing
            logging.error(f"Error processing students in lecture: {str(e)}")  # Log the error

    def _encode_student_images(self, student_id, student_path):
        """
        Load each image, encode it, and remove the image after encoding.

        Parameters:
        - student_id: The ID of the student (folder name).
        - student_path: Path to the folder with the student's images.
        """
        encodings = []  # List to hold all encodings for this student
        processed_images = self._load_processed_images()  # Load the list of processed images

        try:
            for img_file in os.listdir(student_path):  # Loop through each image in the student's folder
                if not self._is_valid_image_type(img_file):
                    continue  # Skip file that are not of the specified types
                img_path = os.path.join(student_path, img_file)  # Get the full path to the image

                if img_path in processed_images:  # Check if the image has already been processed
                    logging.info(f"Skipping already processed image: {img_path}")  # Log the action
                    continue  # Go to the next image

                encoding, cropped_face = self._encode_image(img_path)  # Try to encode the image
                if encoding is not None:  # Check if encoding was successful
                    encodings.append(encoding)  # Add the encoding to the list
                    self._save_processed_image(img_path)  # Mark the image as processed
                    self._save_cropped_image(cropped_face, img_path)  # Save the cropped face image
                    self._remove_image(img_path)  # Delete the original image

            if encodings:  # Check if there are any encodings
                self.student_data[student_id] = encodings  # Save the encodings in the student data
            else:
                logging.warning(f"No encodings found for student {student_id}.")  # Log the error if no encodings found
        except Exception as e:  # If there is an error during encoding
            logging.error(f"Error encoding images for student {student_id}: {str(e)}")  # Log the error

    def _is_valid_image_type(self, image_file):
        """
        Check if the file is one of the valid image types (e.g., 'jpeg', 'jpg', 'png').

        Parameters:
        - image_file: The name of the image file.

        Returns:
        - True if the file is a valid type; False otherwise.
        """
        return any(image_file.lower().endswith(ext) for ext in self.image_types)

    def _encode_image(self, img_path):
        """
        Encode the given image.

        Parameters:
        - img_path: Path to the image to encode.

        Returns:
        - encoding: The face encoding if successful, None otherwise.
        - cropped_face: The cropped face image if successful, None otherwise.
        """
        try:
            image = face_recognition.load_image_file(img_path)  # Load the image
            face_locations = face_recognition.face_locations(image)  # Find face locations in the image

            if not face_locations:  # Check if no faces were found
                logging.warning(f"No faces found in image {img_path}.")  # Log the error
                return None, None  # Return None if no faces found

            if len(face_locations) > 1:  # Check if multiple faces were found
                logging.warning(f"Multiple faces found in image {img_path}.")  # Log the error
                return None, None  # Return None if multiple faces found

            face_encoding = face_recognition.face_encodings(image, face_locations)[0]  # Get the face encoding
            cropped_face = self._crop_face(image, face_locations[0])  # Crop the face from the image
            return face_encoding, cropped_face  # Return the encoding and cropped face
        except Exception as e:  # If there is an error during encoding
            logging.error(f"Error encoding image {img_path}: {str(e)}")  # Log the error
            return None, None  # Return None if encoding failed

    def _crop_face(self, image, face_location):
        """
        Crop the face from the image.

        Parameters:
        - image: The original image.
        - face_location: The location of the face to crop.

        Returns:
        - cropped_face: The cropped face image.
        """
        top, right, bottom, left = face_location  # Unpack face location
        cropped_face = image[top:bottom, left:right]  # Crop the face area from the image
        return cv2.resize(cropped_face, self.CROPPED_IMAGE_SIZE)  # Resize and return the cropped face

    def _load_processed_images(self):
        """
        Load the list of processed images from JSON file.

        Returns:
        - processed_images: A set of paths to processed images.
        """
        try:
            if os.path.exists(self.processed_file):  # Check if the processed file exists
                with open(self.processed_file, 'r') as f:  # Open the JSON file for reading
                    processed_images = json.load(f)  # Load the processed images
                return set(processed_images)  # Return as a set for faster lookup
            return set()  # Return empty set if file does not exist
        except Exception as e:  # If there is an error loading processed images
            logging.error(f"Error loading processed images: {str(e)}")  # Log the error
            return set()  # Return empty set on error

    def _save_processed_image(self, img_path):
        """
        Save a processed image path to the JSON file.

        Parameters:
        - img_path: The path of the image that has been processed.
        """
        processed_images = self._load_processed_images()  # Load current processed images
        processed_images.add(img_path)  # Add the new processed image
        try:
            with open(self.processed_file, 'w') as f:  # Open the JSON file for writing
                json.dump(list(processed_images), f)  # Save the updated list
        except Exception as e:  # If there is an error saving the processed image
            logging.error(f"Error saving processed image: {str(e)}")  # Log the error

    def _save_cropped_image(self, cropped_face, img_path):
        """
        Save the cropped face image.

        Parameters:
        - cropped_face: The cropped face image to save.
        - img_path: The original image path (used to generate a filename).
        """
        try:
            filename = os.path.splitext(os.path.basename(img_path))[0]  # Get the base name without extension
            save_path = os.path.join(self.cropped_faces_dir, f"{filename}_cropped.jpg")  # Create save path
            cv2.imwrite(save_path, cropped_face)  # Save the cropped face image
        except Exception as e:  # If there is an error saving the cropped image
            logging.error(f"Error saving cropped image: {str(e)}")  # Log the error

    def _remove_image(self, img_path):
        """
        Remove the original image file.

        Parameters:
        - img_path: The path of the image to remove.
        """
        try:
            os.remove(img_path)  # Delete the image file
            logging.info(f"Removed image: {img_path}")  # Log successful removal
        except Exception as e:  # If there is an error removing the image
            logging.error(f"Error removing image {img_path}: {str(e)}")  # Log the error

    def _save_encodings(self):
        """
        Save the student data encodings to a pickle file.
        """
        try:
            with open(self.output_file, 'wb') as f:  # Open the output file for writing
                pickle.dump(self.student_data, f)  # Save the student data
            logging.info(f"Saved encodings to {self.output_file}")  # Log successful save
        except Exception as e:  # If there is an error saving the encodings
            logging.error(f"Error saving encodings: {str(e)}")  # Log the error


# If this file is run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # Set up logging to show info messages
    encoder = ImageEncoder()  # Create an instance of ImageEncoder
    encoder.run()  # Start the encoding process