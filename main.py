import pickle  # For loading and saving data
import face_recognition  # For recognizing faces in images
import cv2  # For working with video and images
import numpy as np  # For handling arrays (lists of numbers)

# Load face encodings from the facial_data.pkl file
try:
    # Open the file 'facial_data.pkl' in read-binary mode
    with open('facial_data.pkl', 'rb') as f:
        # Load the data from the file into a variable called 'encoded_faces'
        encoded_faces = pickle.load(f)
except Exception as e:
    # If there's an error, print the error message
    print(f"Error loading facial_data.pkl: {e}")
    # Stop the program if the file can't be loaded
    exit()

# Prepare lists to store known face encodings and names
known_face_encodings = []  # This will hold the face patterns
known_face_names = []  # This will hold the names of the people

# Loop through each key-value pair in the loaded data
for key, value in encoded_faces.items():
    # Add the name (key) to the list of known names
    known_face_names.append(key)
    # Add the face patterns (value) to the list of known encodings
    known_face_encodings.extend(value)  # Make sure all patterns are added

# Get a reference to webcam #0 (the default camera on the computer)
video_capture = cv2.VideoCapture(0)

# Initialize some variables to keep track of face information
face_locations = []  # This will store where faces are located in the video
face_encodings = []  # This will store the face patterns found in the video
face_names = []  # This will store names of recognized faces
process_this_frame = True  # To help with processing every other frame

# Start an infinite loop to keep checking the video feed
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()  # Get a new image from the webcam
    if not ret:
        # If we didn't get a frame, print a message and break the loop
        print("Failed to grab frame")
        break

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize the frame to make it smaller and faster to process
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image color from BGR (used by OpenCV) to RGB (used by face_recognition)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Find all faces and their patterns in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Reset the list of names for this frame
        face_names = []
        for face_encoding in face_encodings:
            # Check if the found face matches any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            if len(matches) > 0:
                # Find the best match with the smallest distance
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)  # Get the index of the closest match

                # Check if the best match is valid
                if matches[best_match_index]:  # If it's a match
                    name = known_face_names[best_match_index]  # Get the name of the match
                else:
                    name = "Unknown"  # If no match, say it's unknown
            else:
                name = "Unknown"  # If no faces found, say it's unknown

            # Add the name to the list
            face_names.append(name)

    # Flip the flag to process the next frame
    process_this_frame = not process_this_frame

    # Draw boxes and names on the video frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was smaller
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face using a red color (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX  # Set the font style for the text
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)  # Add the name text

    # Show the video frame with drawn boxes and names
    cv2.imshow('Video', frame)

    # If the user presses 'q', quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam when done
video_capture.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
