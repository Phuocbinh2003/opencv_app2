import streamlit as st
import cv2 as cv
import numpy as np
from character_segmentation import process_image  # Import hàm từ tệp phân đoạn ký tự


st.title("Ứng dụng Xử lý Hình ảnh")


image_url = "ndata96.jpg"  


image_response = requests.get(image_url)
uploaded_file = image_response.content

original, binary, dilated, dist_transform, img_markers = process_image(uploaded_file)


st.subheader("Ảnh gốc")
st.image(original, channels="BGR")


st.subheader("Ảnh nhị phân")
st.image(binary, use_column_width=True, clamp=True, channels="GRAY")


st.subheader("Ảnh sau khi mở rộng")
st.image(dilated, use_column_width=True, clamp=True, channels="GRAY")


st.subheader("Distance Transform")
st.image(dist_transform, use_column_width=True, clamp=True, channels="GRAY")


st.subheader("Ảnh Watershed Segmentation")
st.image(img_markers, channels="BGR")
