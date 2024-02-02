# Import necessary libraries
import cv2
import time
import face_recognition
import os
import numpy as np
from datetime import datetime

# Path to the folder containing images for training
path = 'ImageBasic'
images = []
classNames = []
myList = os.listdir(path)

# Loop through the image files and append images and corresponding class names to lists
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Print the list of class names
print(classNames)


# Function to encode facial features from training images
def findencodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# Generate encoded representations for training images
encoded_train_images = findencodings(images)
print('Encoding of training image is done')

# Dictionary to store the last entry time for each name
lastEntryTime = {}


# Function to mark attendance in the CSV file
def markAttendance(name):
    # Open the CSV file for reading and writing
    with open('Attendance.csv', 'r+') as f:
        # Read existing data from the CSV file
        myDataList = f.readlines()
        nameList = []

        # Extract existing names from the data
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        # Check if the name is not in the list or if it is and 5 seconds have passed since the last entry
        if name not in nameList or (
                name in nameList and name in lastEntryTime and time.time() - lastEntryTime[name] > 5):
            # Get the current time
            now = datetime.now()
            dt = now.strftime('%H:%M:%S')

            # Write the new entry to the CSV file
            f.writelines(f'\n{name},{dt}')

            # Update the last entry time for the name
            lastEntryTime[name] = time.time()


# Open the webcam
cap = cv2.VideoCapture(0)

# Main loop to capture video frames and perform face recognition
while True:
    # Read a frame from the webcam
    success, img = cap.read()

    # Resize the frame and convert it to RGB format
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Locate faces in the frame and encode their features
    Frame_location = face_recognition.face_locations(imgS)
    Frame_encoding = face_recognition.face_encodings(imgS, Frame_location)

    # Iterate through the found faces
    for encodeFace, faceLoc in zip(Frame_encoding, Frame_location):
        # Compare the face features with the features from the training images
        matches = face_recognition.compare_faces(encoded_train_images, encodeFace)
        faceDis = face_recognition.face_distance(encoded_train_images, encodeFace)
        print(matches)
        print(faceDis)

        # Find the index of the best match
        matchIndex = np.argmin(faceDis)

        # Check if the match is below a certain threshold
        if faceDis[matchIndex] < 0.55:
            # Get the name corresponding to the best match
            name = classNames[matchIndex].upper()
            print(name)

            # Extract face location coordinates
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            # Draw a rectangle around the face and display the name
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Mark attendance for the recognized name
            markAttendance(name)
        else:
            # If the match is below the threshold, consider the person as 'Unknown'
            name = 'Unknown'
            print(name)

            # Extract face location coordinates
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            # Draw a rectangle around the face and display the name as 'Unknown'
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Mark attendance for the 'Unknown' person
            markAttendance(name)

    # Display the frame with the recognized faces
    cv2.imshow('Webcam', img)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
