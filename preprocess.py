import cv2
import numpy as np
from utils import \
    opencv_resize, get_receipt_contour, wrap_perspective, contour_to_rect, remove_noise, bw_scanner, correct_skew
from PIL import Image

def preprocess_image_cu (image):
    #Downscale image.
    #Finding receipt contour is more efficient on a small image
    resize_ratio = 500 / image.shape[0]
    original = image.copy()
    image = opencv_resize(image, resize_ratio)

    # Convert to grayscale for further processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect white regions
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred, rectKernel)
    edged = cv2.Canny(dilated, 50, 125, apertureSize=3)

    #Detect all contours in Canny-edged image
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get 10 largest contours
    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]
    receipt_contour = get_receipt_contour(largest_contours)
    if receipt_contour is not None and cv2.contourArea(receipt_contour) < 5000:
        receipt_contour = None

    if receipt_contour is not None:
      scanned = wrap_perspective(original.copy(), contour_to_rect(receipt_contour, resize_ratio))
    else:
      scanned = original.copy()

    #remove noise
    # scanned = remove_noise(scanned)

    # #Threshold image
    # result = bw_scanner(scanned)

    #final 
    # kernel = np.ones((2,2),np.uint8)
    # eroded_image = cv2.erode(result, kernel, iterations = 1)
    angle, final = correct_skew(scanned.copy())

    return Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))

def preprocess_image_moi (image):
    #Downscale image.
    #Finding receipt contour is more efficient on a small image
    resize_ratio = 500 / image.shape[0]
    original = image.copy()
    image = opencv_resize(image, resize_ratio)

    # Convert to grayscale for further processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect white regions
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred, rectKernel)
    edged = cv2.Canny(dilated, 50, 125, apertureSize=3)

    #Detect all contours in Canny-edged image
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get 10 largest contours
    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]
    receipt_contour = get_receipt_contour(largest_contours)
    if receipt_contour is not None and cv2.contourArea(receipt_contour) < 5000:
        receipt_contour = None

    if receipt_contour is not None:
      scanned = wrap_perspective(original.copy(), contour_to_rect(receipt_contour, resize_ratio))
    else:
      scanned = original.copy()
    angle, scanned = correct_skew(scanned.copy())

    w = scanned.shape[1]
    h = scanned.shape[0]
    w1 = int(w*0.05)
    w2 = int(w*0.95)
    h1 = int(h*0.05)
    h2 = int(h*0.95)
    ROI = scanned[h1:h2, w1:w2]  # 95% of center of the image
    threshold = np.mean(ROI) * 0.9  # % of average brightness

    gray = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    edged = 255 - cv2.Canny(blurred, 100, 150, apertureSize=7)

    thresh, final = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    return Image.fromarray(cv2.cvtColor(scanned, cv2.COLOR_BGR2RGB))