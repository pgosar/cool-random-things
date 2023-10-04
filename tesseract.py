import cv2
import pytesseract
import numpy as np
import shutil
from PIL import Image

init = "temp.png"
filename = "process.png"
pre_processor = "thresh"
shutil.copy(init, "process.png")


def is_skewed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is not None:
        _, theta = lines[0][0]
        angle_degrees = np.degrees(theta)
        return abs(angle_degrees) > 1.0
    else:
        return False


def contrast_stretching(image):
    min_pixel_value = np.min(image)
    max_pixel_value = np.max(image)
    stretched = cv2.convertScaleAbs(image, alpha=255.0 / (max_pixel_value - min_pixel_value),
                                    beta=-min_pixel_value * 255.0 / (max_pixel_value - min_pixel_value))
    return stretched


def histogram_equalization(image):
    equ = cv2.equalizeHist(image)
    return equ


def correct_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is not None:
        _, theta = lines[0][0]
        angle_degrees = np.degrees(theta)
        rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D(
            (image.shape[1] // 2, image.shape[0] // 2), angle_degrees, 1), (image.shape[1], image.shape[0]))
        return rotated
    else:
        return image


img = cv2.imread(filename)
if is_skewed(img):
    gray = correct_skew(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = contrast_stretching(gray)
# gray = histogram_equalization(gray)
if "pre_processor" == "thresh":
    cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
if "pre_processor" == "blur":
    cv2.medianBlur(gray, 3)

cv2.imwrite("temp.png", gray)
text = pytesseract.image_to_string(Image.open(filename))
print(text)
