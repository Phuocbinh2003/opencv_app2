import streamlit as st
import cv2 as cv
import numpy as np
import requests
from matplotlib import pyplot as plt

# Hàm xử lý hình ảnh
def process_image(image):
    # Chuyển đổi sang ảnh xám
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Áp dụng ngưỡng hóa Otsu
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Mở rộng ảnh
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv.dilate(binary, kernel, iterations=1)

    # Khoảng cách transform
    dist_transform = cv.distanceTransform(dilated, cv.DIST_L2, 5)
    
    # Ngưỡng hóa distance transform
    _, sure_fg = cv.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Tìm background chắc chắn
    sure_bg = cv.dilate(dilated, kernel, iterations=2)

    # Vùng chưa biết
    unknown = cv.subtract(sure_bg, sure_fg)

    # Tạo markers cho watershed
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Áp dụng thuật toán Watershed
    img_markers = image.copy()
    cv.watershed(img_markers, markers)
    img_markers[markers == -1] = [0, 0, 255]

    # Tìm các contour
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Vẽ bounding box
    char_id = 1
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if h > 7 and w > 7:  # Lọc những contour có kích thước phù hợp
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return image, binary, dilated, dist_transform, img_markers

st.title("Ứng dụng Xử lý Hình ảnh")

# Tải ảnh từ GitHub
image_url = "https://github.com/Phuocbinh2003/opencv_app2/raw/main/ndata96.jpg"
image_response = requests.get(image_url)
nparr = np.frombuffer(image_response.content, np.uint8)
image = cv.imdecode(nparr, cv.IMREAD_COLOR)

if image is not None:
    # Xử lý ảnh
    original, binary, dilated, dist_transform, img_markers = process_image(image)

    # Hiển thị ảnh gốc
    st.subheader("Ảnh gốc")
    st.image(cv.cvtColor(original, cv.COLOR_BGR2RGB))

    # Hiển thị ảnh nhị phân
    st.subheader("Ảnh nhị phân")
    st.image(binary, channels="GRAY")

    # Hiển thị ảnh sau khi mở rộng
    st.subheader("Ảnh sau khi mở rộng")
    st.image(dilated, channels="GRAY")

    # Hiển thị distance transform
    st.subheader("Distance Transform")
    dist_transform_normalized = cv.normalize(dist_transform, None, 0, 1, cv.NORM_MINMAX)
    st.image(dist_transform_normalized, channels="GRAY")

    # Hiển thị ảnh với watershed markers
    st.subheader("Ảnh Watershed Segmentation")
    st.image(cv.cvtColor(img_markers, cv.COLOR_BGR2RGB))

    # Hiển thị ảnh với bounding box xung quanh các ký tự
    st.subheader("Ảnh với bounding box xung quanh các ký tự")
    st.image(cv.cvtColor(original, cv.COLOR_BGR2RGB))

else:
    st.error("Không thể tải ảnh từ URL.")
