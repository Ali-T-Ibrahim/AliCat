import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# setting path to project directory 
# initialize the categories (target variables for image classification)
CATDIR = r"C:\Users\Ali--\Desktop\Machine Learning Pytorch\Alicat\CatImages"
CATEGORIES = ["coosa", "hobbes"]
IMG_SIZE = 244

cat_image_data = []

def create_image_data():
    for category in CATEGORIES:     # loop used to iterate through every folder of images
        path = os.path.join(CATDIR, category)
        label_index = CATEGORIES.index(category)
        for img in os.listdir(path):    # iterates through images in each path individually
            try:    # filters out all the images that dont get read by pandas properly for any reason
                image = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                cat_image_data.append([resized, label_index])
            except Exception as e:
                pass # not proper practice to pass but not interested in retrieving every single image
    return cat_image_data
                
        
        
        
        
        
        
