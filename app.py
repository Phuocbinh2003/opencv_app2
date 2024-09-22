import streamlit as st
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Hàm để xử lý ảnh
def process_image(uploaded_file):
    # Đọc ảnh
    img = cv.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv.IMREAD_COLOR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Otsu thresholding
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Dilation
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv.dilate(binary, kernel, iterations=1)

    # Distance Transform
    dist_transform = cv.distanceTransform(dilated, cv.DIST_L2, 5)

    # Thresholding the distance transform
    _, sure_fg = cv.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Background
    sure_bg = cv.dilate(dilated, kernel, iterations=2)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Markers for watershed
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed segmentation
    img_markers = img.copy()
    cv.watershed(img_markers, markers)
    img_markers[markers == -1] = [0, 0, 255]  # Red boundary

    return img, binary, dilated, dist_transform, img_markers

# Tiêu đề ứng dụng
st.title("Ứng dụng Xử lý Hình ảnh")

# Tải lên ảnh
uploaded_file = st.file_uploader("Chọn một tệp hình ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Xử lý ảnh
    original, binary, dilated, dist_transform, img_markers = process_image(uploaded_file)

    # Hiển thị ảnh gốc
    st.subheader("Ảnh gốc")
    st.image(original, channels="BGR")

    # Hiển thị ảnh nhị phân
    st.subheader("Ảnh nhị phân")
    st.image(binary, use_column_width=True, clamp=True, channels="GRAY")

    # Hiển thị ảnh sau khi mở rộng
    st.subheader("Ảnh sau khi mở rộng")
    st.image(dilated, use_column_width=True, clamp=True, channels="GRAY")

    # Hiển thị Distance Transform
    st.subheader("Distance Transform")
    st.image(dist_transform, use_column_width=True, clamp=True, channels="GRAY")

    # Hiển thị ảnh Watershed
    st.subheader("Ảnh Watershed Segmentation")
    st.image(img_markers, channels="BGR")

