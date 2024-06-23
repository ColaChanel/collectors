import streamlit as st
import torch
from PIL import Image
from io import BytesIO
import numpy as np
from ultralytics import YOLO

# Загрузка COCO-предобученной модели YOLOv8n
model = YOLO("runs/runs/detect/train/weights/best.pt")


# Функция для детектирования объектов на изображении
def detect_objects(image):
    img = Image.open(image)
    results = model(img)
    detected_objects = results[0]
    annotated_image = detected_objects.plot()
    return annotated_image


# Создание стримлит приложения
def main():
    st.title("Обнаружение объектов на изображении с помощью YOLOv8n")
    uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Загруженное изображение", use_column_width=True)

        if st.button("Обнаружить объекты"):
            result_image = detect_objects(uploaded_image)
            st.image(result_image, caption="Обнаруженные объекты", use_column_width=True, channels='BGR')


if __name__ == '__main__':
    main()
