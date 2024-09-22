import streamlit as st
import cv2 as cv
import requests  # Import the requests library
import numpy as np
from character import process_image

st.title("Ứng dụng Xử lý Hình ảnh")

# Load image from GitHub
image_url = "https://github.com/Phuocbinh2003/opencv_app2/raw/main/ndata96.jpg"
image_response = requests.get(image_url)
nparr = np.frombuffer(image_response.content, np.uint8)
image = cv.imdecode(nparr, cv.IMREAD_COLOR)

if image is not None:
    # Process image
    original, binary, dilated, dist_transform, img_markers = process_image(image)

    # Display original image
    st.subheader("Ảnh gốc")
    st.image(cv.cvtColor(original, cv.COLOR_BGR2RGB))

    # Display binary image
    st.subheader("Ảnh nhị phân")
    st.image(binary, channels="GRAY")

    # Display dilated image
    st.subheader("Ảnh sau khi mở rộng")
    st.image(dilated, channels="GRAY")

    # Display distance transform
    st.subheader("Distance Transform")
    st.image(dist_transform, channels="GRAY")

    # Display watershed segmented image
    st.subheader("Ảnh Watershed Segmentation")
    st.image(cv.cvtColor(img_markers, cv.COLOR_BGR2RGB))
else:
    st.error("Không thể tải ảnh từ URL.")
