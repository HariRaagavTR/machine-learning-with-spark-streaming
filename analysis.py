#!/usr/bin/env python3
import pickle
import numpy as np
import cv2
from skimage.color import rgb2gray
from scipy.ndimage import median_filter

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

MODEL_FOLDER = './model'

def preprocess(image: np.ndarray):
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

def createNewModel(modelType):
    """
    Function that creates a new model instance depending on the modelType argument.
    Args:
        modelType (str): Specifies the model name.
    Returns:
        The respective model reference.
    """
    return {
        'NBClassifier': MultinomialNB(),
        'LRClassifier': LogisticRegression(),
        'SVMClassifier': SGDClassifier()
    } [modelType]
    
def train(X, Y, modelType):
    """
    Function that doesn the following:
    1. Create/Load machine learning model.
    2. Partially train the model using batch data.
    3. Save the model.

    Args:
        X (numpy.array): Feature array, ie, image array.
        Y (numpy.array): Label array.
        modelType (str): Specifies the model name.
    Returns:
        N/A.
    """
    try:
        model = pickle.load(open(MODEL_FOLDER + '/' + modelType + '.sav', 'rb'))
    except:
        print('Info: Saved model does not exist. Creating new ' + modelType + ' model.')
        model = createNewModel(modelType)
    
    model.partial_fit(X, Y)
    print('Info: Partially trained ' + modelType + ' model.')
    
    pickle.dump(model, open(MODEL_FOLDER + '/' + modelType + '.sav', 'wb'))
    print('Info: Saved ' + modelType + ' model @ ' + MODEL_FOLDER + '/' + modelType + '.sav')

def test(X, Y, modelType):
    """
    Function that tests the saved model using unseen data.
    
    Args:
        X (numpy.array): Feature array, ie, image array.
        Y (numpy.array): Label array.
        modelType (str): Specifies the model name.
    Returns:
        N/A.
    """
    try:
        model = pickle.load(open(MODEL_FOLDER + '/' + modelType + '.sav', 'rb'))
        print('Batch Accuracy:', model.score(X, Y))
    except:
        print('Error: Unable to test. Saved model does not exist.')
        return