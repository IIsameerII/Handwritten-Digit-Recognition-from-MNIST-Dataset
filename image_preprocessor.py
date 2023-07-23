import cv2
import numpy as np
from PIL import Image

def extract_digits(image):
    """Takes numpy image data from user and takes the digit 

    Args:
        image: A numpy array
        
    Returns:
        List of individual digits from the canvas
    """

    # This variable has all the digits stored in it once countors are formed
    snip_digits = []

    # Preprocess the image
    thresh = preprocess_image(image)

    # Find contours in the preprocessed image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Filter contours based on area and aspect ratio (assuming digits are approximately square)
    # filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10 and 0.2 <= cv2.arcLength(cnt, True) / cv2.contourArea(cnt) < 0.8]

    # Sort contours from left to right
    filtered_contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

    # Extract and save individual digits
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        digit = thresh[y:y+h, x:x+w]
        digit = image_padding(digit,w,h)
        snip_digits.append(digit)

    return snip_digits
        

def preprocess_image(image):
    """Takes numpy image data from user and grayscale's and applies threshold

    Args:
        image: A numpy array
        
    Returns:
        thresh: A numpy array
    """

    # Read the image and convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to make the digits stand out
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return thresh

def image_padding(image,w,h,margin=10):
    """Takes numpy image data, width and height of the image and margin (whitespace in the edges) and applies margin to the image

    Args:
        image: A numpy array
        w: width of image (int)
        h: height of image (int)
        margin: margin for the image at the end (int)
        
    Returns:
        padded_image: A numpy array
    """
    
    # if width is greater than height
    if w>h:
        diff = w-h
        half_diff = int(round(diff/2))
        padded_image = cv2.copyMakeBorder(image, half_diff+margin, half_diff+margin, margin, margin, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded_image
    
    # if height is greater than width
    elif h>w:
        diff = h-w
        half_diff = int(round(diff/2))
        padded_image = cv2.copyMakeBorder(image, margin, margin, half_diff+margin, half_diff+margin, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded_image
    

def invert_colors_opencv(image):
    # Convert the image to a NumPy array
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Invert colors by subtracting each pixel value from 255 (assuming uint8 datatype)
    inverted_image = 255 - image

    return inverted_image





    