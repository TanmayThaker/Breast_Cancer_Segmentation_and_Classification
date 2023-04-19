import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pyngrok import ngrok
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import matplotlib
from joblib import Memory
matplotlib.use('TkAgg')
st.set_option('deprecation.showfileUploaderEncoding',False)

# Set up a cache directory
cachedir = './model_cache'
memory = Memory(cachedir, verbose=0)

def post_process(segmented_image):
    segmented_array = segmented_image.squeeze().astype(np.uint8)
    print(segmented_array.shape)  # prints (height, width, channels)
    # Use a larger kernel size
    kernel = np.ones((7, 7), np.uint8)

    # Apply morphological closing and opening operations
    closed = cv2.morphologyEx(segmented_array, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened

@memory.cache
def predict_image(img):
    localize =   load_model('C:\\Users\\tanma\\Downloads\\Breast_Cancer_Final\\unet.h5')
    # classifier = load_model('C:\\Users\\tanma\\Downloads\\Breast_Cancer_Final\\valid_classifier_resnet101_version3.h5')
    classifier = load_model('C:\\Users\\tanma\\Downloads\\Breast_Cancer_Final\\valid_classifier_resnet101_version3.h5')
    #print(localize.summary())
    testX = []
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    testX.append(cv2.resize(gray_image, (128,128)))
 

    testX = np.array(testX)
    testX = testX.astype('float32') / 255.0
    testX = testX.reshape(1, 128, 128, 1)

    # Localizing the cancer using the segmentation model
    predY = localize.predict(testX)
    print("PredY shape: ",predY.shape)
    segmented_image = post_process(predY)
    print(type(predY))
    

    segmented_image = predY.astype('float32') * 255.0
    segmented_image = segmented_image.squeeze().astype(np.uint8)
    cv2.imwrite("Segmented_image.png",segmented_image)
    segmented_image = Image.fromarray(segmented_image)
    with col2:
        st.image(segmented_image, caption="Segmented Image",use_column_width = True,channels="L")

    predY = np.stack((predY,)*3, axis=-1)  # stack the array
    predY = np.squeeze(predY,axis=3)  # remove extra dimension
    print("PredY shape before classification is: ",predY.shape)
    # Classifying the cancer using the classifier model
    pred_label = classifier.predict(predY)
    print("Prediction labels are: ",pred_label)
    return int(np.argmax(pred_label, axis = 1))

st.write('''# Breast Cancer Classification''')

file = st.file_uploader("Please upload an image: ",type=["jpg","png"])
if file is None:
    st.text("Please upload an image file")
else:
    col1, col2 = st.columns(2)
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    # decode numpy array as opencv image
    opencv_image = cv2.imdecode(file_bytes, 1)
    # display opencv image using streamlit
    with col1:
        st.image(opencv_image, channels="BGR",use_column_width=True)
    prediction=predict_image(opencv_image)
    info = [
    'benign'   ,  
    'normal'   ,  
    'malignant',  
    ]
    string="This image most likely is: "+info[prediction]
    st.success(string)