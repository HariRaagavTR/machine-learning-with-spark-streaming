#!/usr/bin/env python3
import numpy as np
import cv2
from skimage.color import rgb2gray
from scipy.ndimage import median_filter

def preprocess(image):
    """
    Function that receives an image and performs the following operations:
    1. Convert the image to grayscale.
    2. Resize the image.
    3. Apply median filter to smoothen the image.
    4. Apply cv2.Canny() to retrieve all edges from the image.
    5. Flatten the image to a 1D numpy array.

    Args:
        image (numpy.ndarray): Image to be processed.
    Returns:
        numpy.array: Preprocessed and flattened image.
    """
    try:
        grayScaleImage = rgb2gray(image)
        resizedImage = cv2.resize(grayScaleImage, (256, 256))

        blurredImage = (
            median_filter(resizedImage, (3, 3)) * 255
        ).astype(np.uint8)

        preprocessedImage = cv2.Canny(
            image = blurredImage,
            threshold1 = 100, 
            threshold2 = 200
        ).flatten() // 255
        
        return preprocessedImage
    
    except Exception as error:
        print("Error @ Preprocessor:", error)
        return np.asarray([])
        