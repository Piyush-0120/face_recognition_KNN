import cv2
import numpy as np

cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier(
    "C:\\Users\\LEGION\\Documents\\Jupyter Notebook\\haarcascade_frontalface_alt.xml")
# the above data is about faces

skip = 0
face_data = []
dataset_path = "./data/"
file_name = input("Enter the name of person:")

while True:
    ret, frame = cap.read()

    if ret == False:
        # image not captured, webcam not started,etc.
        continue
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3])

    # Pick the last face(because it is the largest face acc to area(f[2]*f[3]))
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue frame

        # Extract (crop out the required face) :Region of Interest
        offset = 10  # padding of 10 pixels in all directions
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        skip +=1
        if (skip % 10 == 0):
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow("Video Frame", frame)

    # Wait for user input - q, then you will stop the loop
    key_pressed = cv2.waitKey(1) & 0xFF  # 32 bit no. & 8 1's->then we get last 8 bits
    # trying to convert 32 bit to a 8 bit no.
    if key_pressed == ord('q'):  # ascii value of this character
        break

#Convert our face list array into a numpy array
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save data in file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully saved at "+dataset_path+file_name+".npy")

cap.release()
cv2.destroyAllWindows()
