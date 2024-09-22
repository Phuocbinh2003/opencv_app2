import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests  # Make sure to include this import

# Danh sách các URL hoặc đường dẫn tới các hình ảnh
image_urls = [
    "https://github.com/Phuocbinh2003/opencv_app2/raw/main/ndata349.jpg",
    "https://github.com/Phuocbinh2003/opencv_app2/raw/main/ndata235.jpg",
    "https://github.com/Phuocbinh2003/opencv_app2/raw/main/ndata397.jpg",
    "https://github.com/Phuocbinh2003/opencv_app2/raw/main/ndata407.jpg",
    "https://github.com/Phuocbinh2003/opencv_app2/raw/main/ndata360.jpg",
    "https://github.com/Phuocbinh2003/opencv_app2/raw/main/ndata470.jpg",
    "https://github.com/Phuocbinh2003/opencv_app2/raw/main/ndata497.jpg",
    "https://github.com/Phuocbinh2003/opencv_app2/raw/main/ndata592.jpg",
    "https://github.com/Phuocbinh2003/opencv_app2/raw/main/ndata96.jpg",
    "https://github.com/Phuocbinh2003/opencv_app2/raw/main/ndata89.jpg",
]

# Hàm xử lý hình ảnh
def process_image(image):
    # (giữ nguyên phần này như trước)
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
    img_watershed = image.copy()
    cv.watershed(img_watershed, markers)
    img_watershed[markers == -1] = [0, 0, 255]
    
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    characters = []
    char_images = []
    char_id = 1
    image_with_boxes = image.copy()

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if h > 7 and w > 7 and h < 100 and w < 100:
            cv.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
            char_image = binary[y:y+h, x:x+w]  
            characters.append(f"Ký tự {char_id}") 
            char_images.append(char_image)  
            char_id += 1

    return image, binary, dilated, dist_transform, sure_fg, sure_bg, unknown, markers, characters, char_images, img_watershed, image_with_boxes

# Xây dựng ứng dụng
st.title('✨ Phân đoạn ký tự biển số ')

st.divider()

st.sidebar.write("## 📷 Select Image")
selected_image = st.sidebar.selectbox("Choose an image", image_urls)

# Tải ảnh từ URL đã chọn
image_response = requests.get(selected_image)
nparr = np.frombuffer(image_response.content, np.uint8)
img_np = cv.imdecode(nparr, cv.IMREAD_COLOR)

if img_np is not None:
    st.image(img_np, caption='Selected Image.', use_column_width=True)

    # Thực hiện nhận diện biển số bằng Watershed
    if st.button('Detect License Plate'):
        processed_image, binary, dilated, dist_transform, sure_fg, sure_bg, unknown, markers, characters, char_images, img_watershed, image_with_boxes = process_image(img_np)

        st.write("### Processing")

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(17, 17))

        # Hiển thị các kết quả trung gian
        axes[0, 0].imshow(cv.cvtColor(img_np, cv.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 1].imshow(binary, cmap='gray')
        axes[0, 1].set_title('Binarization')
        axes[0, 2].imshow(dilated, cmap='gray')
        axes[0, 2].set_title('Dilated Image')
        axes[1, 0].imshow(dist_transform, cmap='gray')
        axes[1, 0].set_title('Distance Transform')
        axes[1, 1].imshow(sure_fg, cmap='gray')
        axes[1, 1].set_title('Sure Foreground')
        axes[1, 2].imshow(sure_bg, cmap='gray')
        axes[1, 2].set_title('Sure Background')
        axes[2, 0].imshow(unknown, cmap='gray')
        axes[2, 0].set_title('Unknown Region')
        axes[2, 1].imshow(cv.cvtColor(img_watershed, cv.COLOR_BGR2RGB))
        axes[2, 1].set_title('Watershed Segmentation')

        for ax in axes.flatten():
            ax.axis('off')

        st.pyplot(fig)

        # Hiển thị các ký tự phát hiện được
        st.subheader("Các ký tự phát hiện được")
        cols = st.columns(len(char_images))
        for idx, char_img in enumerate(char_images):
            with cols[idx]:
                st.image(char_img, caption=f"Ký tự {idx + 1}", channels="GRAY")

        # Hiển thị ảnh đã vẽ bounding boxes
        st.subheader("Processed Image with Bounding Boxes")
        st.image(image_with_boxes, channels="BGR")
else:
    st.error("Không thể tải ảnh từ URL.")
