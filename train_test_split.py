import os
import glob
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from sklearn.model_selection import train_test_split



data_folder = ["image_training","image_testing"]
for folder in data_folder :
    if not os.path.exists(folder):
        os.makedirs(folder) 
csv_dat = pd.read_csv("image.csv") 
index = csv_dat["Class"]
train_image,test_image,train_class,test_class = train_test_split(csv_dat,index,test_size=0.2,shuffle =True,stratify =index)
# save file
train_image.to_csv("train_image.csv")
test_image.to_csv("test_image.csv")
for image_path in train_image["Name"]:
    image = cv2.imread(image_path)
    cv2.imwrite(os.path.join("image_training",image_path + ".jpg"),image)
for image_path in test_image["Name"]:
    image = cv2.imread(image_path)
    cv2.imwrite(os.path.join("image_testing",image_path + ".jpg"),image)