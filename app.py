import cv2
import streamlit as st
import numpy as np
from keras.models import model_from_json
import tempfile

st.set_page_config(page_title="FYP1",page_icon="😀")
st.title("Facial Expressions Recognition System")

option = st.sidebar.selectbox(
    "Select an option",
    ("Real-time Recognition", "Upload Video")
)
FRAME_WINDOW = st.image([])
stframe = st.empty()

if option == 'Real-time Recognition':
    start = st.checkbox('Start')
    while start:
        camera = cv2.VideoCapture(0)
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        # load json and create model
        json_file = open('./model/model4.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        emotion_model = model_from_json(loaded_model_json)

        # load weights into new model
        emotion_model.load_weights("./model/model.h5")

        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            FRAME_WINDOW.image(frame)
    else:
        st.write("Stopped")
else:
    uploaded_video = st.file_uploader("Please upload a video.")
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
  
        camera = cv2.VideoCapture(tfile.name)
        while camera.isOpened():
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

            # load json and create model
            json_file = open('./model/model4.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            emotion_model = model_from_json(loaded_model_json)

            # load weights into new model
            emotion_model.load_weights("./model/model.h5")

            face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces available on camera
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            # take each face available on the camera and Preprocess it
            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                # predict the emotions
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                FRAME_WINDOW.image(frame)
                # stframe.image(frame)
    else:
        st.write(" ")