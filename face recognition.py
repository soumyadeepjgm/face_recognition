import numpy as np
import cv2
import face_recognition
import csv
from datetime import datetime
from PIL import Image


# Function to load and convert image to RGB
def load_and_convert_image(file_path):
    try:
        img = Image.open(file_path)
        rgb_img = img.convert('RGB')
        rgb_img = np.array(rgb_img)
        print(f"Loaded image {file_path} with shape: {rgb_img.shape} and dtype: {rgb_img.dtype}")
        return rgb_img
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None


# Capturing video from OpenCV
video_capture = cv2.VideoCapture(0)

# Attribute training paths
Soumyadeep_image_path = "D:\\PROJECT ALL\\FACE RECOGNITION\\soumyadeep1.jpg"
Soumyadeep_image = load_and_convert_image(Soumyadeep_image_path)
if Soumyadeep_image is None:
    exit()

# Encoding faces
try:
    Soumyadeep_encoding = face_recognition.face_encodings(Soumyadeep_image)[0]
except IndexError:
    print("No face detected in the image.")
    exit()

known_face_encodings = [
    Soumyadeep_encoding,
]

# Denoting names of the given attributes
known_face_names = [
    "Soumyadeep Mandal",
]

people = known_face_names.copy()

face_locations = []  # used to save the face location
face_names = []  # names of the faces

current_date = datetime.now().strftime("%Y-%m-%d")  # for determining the date of capture

# Opening a CSV file in write mode
with open(current_date + '.csv', 'w+', newline='') as f:
    lnwriter = csv.writer(f)

    while True:
        _, frame = video_capture.read()  # taking the video input
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # decreasing the size of the input from the webcam
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # converting into RGB

        face_locations = face_recognition.face_locations(rgb_small_frame)  # detecting faces
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # encoding the faces

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)  # comparing faces
            name = ""  # after recognition, this variable will contain the name of the face
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            if name in people:
                people.remove(name)  # removing the name from the list
                print(people)
                current_time = datetime.now().strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])  # entering the name and time in the CSV file

        cv2.imshow("Present People System", frame)  # showing output

        if cv2.waitKey(1) & 0xFF == ord('q'):  # exit on pressing 'q'
            break

# Release the capture and close any OpenCV windows
video_capture.release()
cv2.destroyAllWindows()