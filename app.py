import cv2
import numpy as np
from superpoint.datasets.synthetic_shapes import SyntheticShapes
from utils import plot_imgs
import streamlit as st
from PIL import Image

# Cấu hình dữ liệu
config = {
    'primitives': 'all',
    'on-the-fly': True,
    'preprocessing': {'resize': [120, 160], 'blur_size': 21}
}
dataset = SyntheticShapes(**config)
data = dataset.get_test_set()

# Hàm vẽ keypoint
def draw_keypoints(img, corners, color):
    keypoints = [cv2.KeyPoint(c[1], c[0], 1) for c in np.stack(corners).T]
    return cv2.drawKeypoints(img.astype(np.uint8), keypoints, None, color=color)

# Hàm hiển thị ảnh
def display(d):
    return draw_keypoints(d['image'][..., 0] * 255, np.where(d['keypoint_map']), (0, 255, 0))

# Tạo ứng dụng Streamlit
st.title("SuperPoint Keypoint Visualization")

# Hiển thị ảnh
for i in range(8):
    st.subheader(f"Set {i+1}")
    col1, col2, col3, col4 = st.columns(4)  # Tạo 4 cột để hiển thị 4 ảnh

    for col in [col1, col2, col3, col4]:
        d = next(data)
        img = display(d)

        # Chuyển đổi ảnh từ định dạng OpenCV (BGR) sang định dạng PIL (RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Hiển thị ảnh trong cột
        col.image(pil_img, use_column_width=True)
