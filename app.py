import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load models
face_pbtxt = "models/opencv_face_detector.pbtxt"
face_pb = "models/opencv_face_detector_uint8.pb"
age_prototxt = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_prototxt = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"
MODEL_MEAN_VALUES = [104,117,123]

# Load DNNs
face_net = cv2.dnn.readNet(face_pb, face_pbtxt)
age_net = cv2.dnn.readNet(age_model, age_prototxt)
gender_net = cv2.dnn.readNet(gender_model, gender_prototxt)

# Labels
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Streamlit App
st.set_page_config(layout="centered")
st.title("🧠 Gender and Age Detection App")
st.write("Upload a photo and we'll predict the gender and age of the person.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_cp = img.copy()
    
    # Face Detection
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), MODEL_MEAN_VALUES, True, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    
    results = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.99:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            
            face_img = img_cp[max(0, y1-15):min(y2+15, h-1), max(0, x1-15):min(x2+15, w-1)]
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=True)

            gender_net.setInput(blob)
            gender = gender_list[gender_net.forward()[0].argmax()]
            
            age_net.setInput(blob)
            age = age_list[age_net.forward()[0].argmax()]
            
            label = f"{gender}, {age}"
            cv2.rectangle(img_cp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cp, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            results.append(label)
    
    st.image(cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB), caption="Detected Results", use_column_width=True)

    if results:
        st.success("Detection Complete ✅")
    else:
        st.warning("No faces detected. Try another image.")
