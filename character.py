# character.py

import cv2 as cv
import numpy as np

def process_image(uploaded_file):
    # Dữ liệu ảnh được truyền vào đã là mảng numpy, nên không cần gọi .read()
    img = uploaded_file
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Otsu thresholding
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Dilation to separate characters
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv.dilate(binary, kernel, iterations=1)

    # Distance Transform
    dist_transform = cv.distanceTransform(dilated, cv.DIST_L2, 5)

    # Foreground and background separation
    _, sure_fg = cv.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv.dilate(dilated, kernel, iterations=2)

    # Unknown region
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    img_markers = img.copy()
    cv.watershed(img_markers, markers)
    img_markers[markers == -1] = [0, 0, 255]  # Mark boundaries in red

    return img, binary, dilated, dist_transform, img_markers
