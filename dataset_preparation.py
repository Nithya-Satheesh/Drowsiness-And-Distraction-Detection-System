import dlib
import cv2
import os
import time

# Load the face detector from dlib's pre-trained model
detector = dlib.get_frontal_face_detector()

# Create a VideoCapture object to capture video from the default camera
cap = cv2.VideoCapture(0)

tod="train" #type of data
classes="side_faces" #class type


# Create a directory to save the detected faces
if not os.path.exists(f'{tod}/{classes}'):
    os.makedirs(f'{tod}/{classes}')

# Initialize variables
frame_count = 0
start_time = time.time()

# Counter for the detected faces
count = 903

# Loop through the frames captured from the camera
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    frame_count += 1

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using dlib's face detector
    faces = detector(gray)

    # Loop through the detected faces and save each face as a separate image file
    for face in faces:
        # Extract the coordinates of the face rectangle
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        # Crop the detected face from the frame
        face_image = frame[y1:y2, x1:x2]

        # Save the detected face as a separate image file
        cv2.imwrite(f'{tod}/{classes}/face_{count}.jpg', face_image)

        # Increment the counter for the detected faces
        count += 1

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the frame with the detected faces
    cv2.imshow('Frame', frame)

    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        # Calculate the frames per second (FPS)
        fps = frame_count / elapsed_time
        print("Frames per second:", fps)

        # Reset the frame count and start time for the next second
        frame_count = 0
        start_time = time.time()
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second:", fps)

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()

