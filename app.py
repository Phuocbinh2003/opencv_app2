import streamlit as st
import cv2 as cv
import numpy as np
import requests

# Hàm xử lý hình ảnh
def process_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv.dilate(binary, kernel, iterations=1)
    
    dist_transform = cv.distanceTransform(dilated, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    sure_bg = cv.dilate(dilated, kernel, iterations=2)
    unknown = cv.subtract(sure_bg, sure_fg)
    
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    img_markers = image.copy()
    cv.watershed(img_markers, markers)
    img_markers[markers == -1] = [0, 0, 255]

    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    characters = []
    char_images = []  # Danh sách ảnh ký tự
    char_id = 1
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if h > 7 and w > 7 and h<100 and w<100:  # Điều chỉnh kích thước tối thiểu
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            char_image = binary[y:y+h, x:x+w]  # Cắt từng ký tự
            characters.append(f"Ký tự {char_id}")  # Tạo tên cho ký tự
            char_images.append(char_image)  # Lưu ảnh ký tự
            char_id += 1

    return image, binary, dilated, dist_transform, img_markers, characters, char_images

st.title("Ứng dụng Xử lý Hình ảnh")

# Tải ảnh từ GitHub
image_url = "https://github.com/Phuocbinh2003/opencv_app2/raw/main/ndata96.jpg"
image_response = requests.get(image_url)
nparr = np.frombuffer(image_response.content, np.uint8)
image = cv.imdecode(nparr, cv.IMREAD_COLOR)

if image is not None:
    # Xử lý ảnh
    original, binary, dilated, dist_transform, img_markers, characters, char_images = process_image(image)

    # Hiển thị ảnh gốc
    st.subheader("Ảnh gốc")
    st.image(cv.cvtColor(original, cv.COLOR_BGR2RGB))

    # Hiển thị các ký tự theo chiều ngang
    st.subheader("Các ký tự phát hiện được")
    cols = st.columns(len(char_images))  # Tạo các cột tương ứng với số ký tự
    for idx, char_img in enumerate(char_images):
        with cols[idx]:
            st.image(char_img, caption=f"Ký tự {idx + 1}", channels="GRAY")

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
