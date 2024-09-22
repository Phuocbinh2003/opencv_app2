import streamlit as st
import cv2 as cv
import numpy as np
import requests

# Hàm xử lý hình ảnh
def process_image(image):
    # Chuyển đổi sang ảnh xám
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Nhị phân hóa
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Mở rộng (dilation)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv.dilate(binary, kernel, iterations=1)

    # Distance Transform
    dist_transform = cv.distanceTransform(dilated, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Xác định sure background và unknown region
    sure_bg = cv.dilate(dilated, kernel, iterations=2)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Gán nhãn cho các markers
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Áp dụng watershed
    img_markers = image.copy()
    cv.watershed(img_markers, markers)
    img_markers[markers == -1] = [0, 0, 255]

    # Tìm contour và tách ký tự
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    characters = []
    char_images = []
    char_id = 1
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if h > 7 and w > 7 and h < 100 and w < 100:  # Điều chỉnh kích thước tối thiểu
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1) 
            char_image = binary[y:y+h, x:x+w]  
            characters.append(f"Ký tự {char_id}") 
            char_images.append(char_image)  
            char_id += 1

    return image, binary, dilated, dist_transform, sure_fg, sure_bg, unknown, img_markers, characters, char_images

# Xây dựng giao diện Streamlit
st.title("Ứng dụng Xử lý Hình ảnh - Quy trình Watershed")

# Tải ảnh từ GitHub
image_url = "https://github.com/Phuocbinh2003/opencv_app2/raw/main/ndata96.jpg"
image_response = requests.get(image_url)
nparr = np.frombuffer(image_response.content, np.uint8)
original_image = cv.imdecode(nparr, cv.IMREAD_COLOR)

if original_image is not None:
    # Xử lý ảnh qua từng bước
    processed_image, binary, dilated, dist_transform, sure_fg, sure_bg, unknown, img_markers, characters, char_images = process_image(original_image.copy())

    # Hiển thị ảnh gốc
    st.subheader("1. Ảnh gốc")
    st.image(cv.cvtColor(original_image, cv.COLOR_BGR2RGB))

    # Hiển thị ảnh nhị phân
    st.subheader("2. Ảnh nhị phân")
    st.image(binary, channels="GRAY")

    # Hiển thị ảnh sau khi mở rộng (Dilation)
    st.subheader("3. Ảnh sau khi mở rộng (Dilation)")
    st.image(dilated, channels="GRAY")

    # Hiển thị Distance Transform
    st.subheader("4. Distance Transform")
    dist_transform_normalized = cv.normalize(dist_transform, None, 0, 1, cv.NORM_MINMAX)
    st.image(dist_transform_normalized, channels="GRAY")

    # Hiển thị Sure Foreground
    st.subheader("5. Sure Foreground")
    st.image(sure_fg, channels="GRAY")

    # Hiển thị Sure Background
    st.subheader("6. Sure Background")
    st.image(sure_bg, channels="GRAY")

    # Hiển thị Unknown Region
    st.subheader("7. Unknown Region")
    st.image(unknown, channels="GRAY")

    # Hiển thị ảnh sau khi áp dụng Watershed
    st.subheader("8. Ảnh sau khi áp dụng Watershed Segmentation")
    st.image(cv.cvtColor(img_markers, cv.COLOR_BGR2RGB))

    # Hiển thị ký tự đã phát hiện
    st.subheader("9. Các ký tự phát hiện được")
    if char_images:
        cols = st.columns(len(char_images))  # Tạo các cột tương ứng với số ký tự
        for idx, char_img in enumerate(char_images):
            with cols[idx]:
                st.image(char_img, caption=f"Ký tự {idx + 1}", channels="GRAY")
    else:
        st.write("Không phát hiện được ký tự nào.")

else:
    st.error("Không thể tải ảnh từ URL.")
