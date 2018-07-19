import os
import glob
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import re

file = pd.read_fwf("list_bbox.txt",skiprows=[0,1],header=None)
data = pd.DataFrame(file)
data.columns =["Path_image","x_1","y_1","x_2","y_2"]
num_image = len(data)
data.insert(1,column="Name",value=np.ones(num_image))
data.insert(2,column="Height",value=np.ones(num_image))
data.insert(3,column="Width",value=np.ones(num_image))
data.insert(4,column="Class",value=np.ones(num_image))
path_images = data["Path_image"]
df = data.copy()

for idx,path in enumerate(path_images):
        image = cv2.imread(path)
        if image is not None:
            height, width = image.shape[:2]
            #height = image.shape[0]
            #width = image.shape[1]
            classes = re.search(r"(?<=_)([a-zA-Z0-9]+)(?=/)",path)
            if classes is not None :
                classes = classes.group(0)
                df["Name"].iloc[idx] = classes + "_" + str(idx)
                df["Height"].iloc[idx] = int(height)
                df["Width"].iloc[idx] = int(width)
                df["Class"].iloc[idx] = classes
                cv2.imwrite(classes + "_" + str(idx) + ".jpg",image)
            if idx % 1000 == 0 :
                print("num :",idx)
image_csv = df[df["Class"]!=1]
image_csv.to_csv("image.csv")