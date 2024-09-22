import cv2 as cv
import numpy as np

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

    # Tìm các contour và vẽ bounding box
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if h > 7 and w > 7:  # Lọc kích thước phù hợp
            cv.rectangle(img_markers, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Vẽ bounding box

    return img, binary, dilated, dist_transform, img_markers
