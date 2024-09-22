import streamlit as st
import cv2 as cv
import numpy as np
import requests
from character import process_image  # Import hàm từ tệp phân đoạn ký tự

# Tiêu đề ứng dụng
st.title("Ứng dụng Xử lý Hình ảnh")

# Tải ảnh từ GitHub
image_url = "https://github.com/Phuocbinh2003/opencv_app2/blob/main/ndata96.jpg"  # Cập nhật đường dẫn đúng

# Đọc ảnh từ URL
image_response = requests.get(image_url)
image_bytes = image_response.content

# Chuyển bytes thành định dạng hình ảnh mà OpenCV có thể xử lý
nparr = np.frombuffer(image_bytes, np.uint8)
image = cv.imdecode(nparr, cv.IMREAD_COLOR)

# Xử lý ảnh
original, binary, dilated, dist_transform, img_markers = process_image(image)

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
