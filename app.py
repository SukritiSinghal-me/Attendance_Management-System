import streamlit as st
import pandas as pd
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import os

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Attendance Management System",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Attendance Management System")
st.write("Using YOLO Person Detection")

# -------------------------------
# Load YOLO Model
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -------------------------------
# Attendance File
# -------------------------------
attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Person", "Date", "Time"])
    df.to_csv(attendance_file, index=False)

# -------------------------------
# Function to Mark Attendance
# -------------------------------
def mark_attendance(person_name):

    df = pd.read_csv(attendance_file)

    today = datetime.now().strftime("%Y-%m-%d")

    already_marked = (
        (df["Person"] == person_name) &
        (df["Date"] == today)
    ).any()

    if not already_marked:

        current_time = datetime.now().strftime("%H:%M:%S")

        new_row = pd.DataFrame({
            "Person":[person_name],
            "Date":[today],
            "Time":[current_time]
        })

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(attendance_file, index=False)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Options")

option = st.sidebar.radio(
    "Choose Input",
    ["Upload Image", "Webcam"]
)

# -------------------------------
# Upload Image
# -------------------------------
if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        image_np = np.array(image)

        results = model(image_np)

        boxes = results[0].boxes

        person_count = 0

        for box in boxes:

            cls = int(box.cls[0])

            if cls == 0:

                person_count += 1

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                cv2.rectangle(
                    image_np,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    image_np,
                    "Person",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,0,0),
                    2
                )

                person_name = f"Person_{person_count}"

                mark_attendance(person_name)

        st.image(image_np,
                 caption="Detected Persons",
                 use_container_width=True)

        st.success(f"Detected {person_count} Person(s)")

# -------------------------------
# Webcam
# -------------------------------
elif option == "Webcam":

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:

        ret, frame = camera.read()

        if not ret:
            st.error("Camera not working")
            break

        results = model(frame)

        boxes = results[0].boxes

        person_count = 0

        for box in boxes:

            cls = int(box.cls[0])

            if cls == 0:

                person_count += 1

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Person {person_count}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,0,0),
                    2
                )

                person_name = f"Person_{person_count}"

                mark_attendance(person_name)

        FRAME_WINDOW.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

    camera.release()

# -------------------------------
# Attendance Table
# -------------------------------
st.subheader("Attendance Record")

attendance = pd.read_csv(attendance_file)

st.dataframe(attendance)

# -------------------------------
# Download Attendance
# -------------------------------
csv = attendance.to_csv(index=False)

st.download_button(
    "Download Attendance",
    csv,
    "attendance.csv",
    "text/csv"
)