import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Blind Assistive System",
    layout="centered"
)

st.title("Blind Assistive System")
st.write("AI Powered Object Detection for Visually Impaired Assistance")

model = YOLO("models/yolov8n.pt")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image_np = np.array(image)

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    with st.spinner("Detecting objects..."):

        results = model(image_np)

        annotated_frame = results[0].plot()

    st.subheader("Detection Results")

    st.image(
        annotated_frame,
        caption="Detected Objects",
        use_container_width=True
    )

    detected_objects = []

    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        detected_objects.append(class_name)

    unique_objects = list(set(detected_objects))

    st.subheader("Detected Objects List")

    if unique_objects:
        for obj in unique_objects:
            st.write(f"• {obj}")
    else:
        st.write("No objects detected.")