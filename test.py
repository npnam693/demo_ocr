from pytesseract import Output
import pytesseract
import argparse
import imutils
import cv2
from preprocess import preprocess_image_moi, preprocess_image_cu
from rembg import remove
from PIL import Image
import numpy as np
from skimage import io


# load the input image
def crop_background(image, grayscale=False):
    """Crop black background only"""
    if isinstance(image, Image.Image):
        img_arr = np.array(image)
    else:
        img_arr = io.imread(image, as_gray=True)

    gray = img_arr[:, :, 0] if img_arr.ndim > 2 else img_arr
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    x, y, w, h = cv2.boundingRect(thresholded)
    output = (gray if grayscale else img_arr)[y:y+h, x:x+w]
    return output

try:
    image = Image.open("tee.png")
    if image is None:
        raise Exception("Error: Unable to load the input image.")
except Exception as e:
    print(str(e))
    exit()
# Use rembg to remove the background
newImg = preprocess_image_moi(np.array(image))
# newImg.show()

# output_image = crop_background(output_image)
# newImg = preprocess_image_moi(np.array(output_image))
newImg.show()
# print(pytesseract.image_to_string(newImg))
# print('-----', pytesseract.image_to_string(output_image))
print('-----', pytesseract.image_to_string(image, config='--oem 1 --psm 6'))
print('-----', pytesseract.image_to_string(image, config='--oem 3 --psm 6'))






    
