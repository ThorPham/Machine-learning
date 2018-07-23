import os
import glob
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import re
import shutil
from sklearn.model_selection import train_test_split

category_name_generate = ['Anorak', 'Blazer', 'Blouse', 'Bomber', 'Button-Down', 'Cardigan', 
                          'Flannel', 'Halter', 'Henley', 'Hoodie', 'Jacket', 'Jersey', 'Parka',
                          'Peacoat', 'Poncho', 'Sweater', 'Tank', 'Tee', 'Top', 'Turtleneck', 'Capris', 
                          'Chinos', 'Culottes', 'Cutoffs', 'Gauchos', 'Jeans', 'Jeggings', 'Jodhpurs', 
                          'Joggers', 'Leggings', 'Sarong', 'Shorts', 'Skirt', 'Sweatpants', 'Sweatshorts',
                          'Trunks', 'Caftan', 'Coat', 'Coverup', 'Dress', 'Jumpsuit', 'Kaftan', 'Kimono', 
                          'Onesie', 'Robe', 'Romper']

# make folder 
#for folder in category_name_generate :
#if not os.path.exists(folder):
#os.makedirs(folder)``` 
        
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

for idx,path in enumerate(path_images[:10]):
        image = cv2.imread(path)
        if image is not None:
            height, width = image.shape[:2]
            #height = image.shape[0]
            #width = image.shape[1]
            classes = re.search(r"(?<=_)([a-zA-Z0-9]+)(?=/)",path)
            if classes is not None :
                classes = classes.group(0)
                if classes in category_name_generate:
                    df["Name"].iloc[idx] = classes + "_" + str(idx)
                    df["Height"].iloc[idx] = int(height)
                    df["Width"].iloc[idx] = int(width)
                    df["Class"].iloc[idx] = classes
                    #cv2.imwrite(os.path.join(classes,classes + "_" + str(idx) + ".jpg"),image)
                    if idx % 1000 == 0 :
                        print("num :",idx)
image_csv = df[df["Class"]!=1]
image_csv.to_csv("image.csv")
#image_csv.to_csv("image.csv")

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
for index,image_path in enumerate(train_image["Path_image"]):
    classes = re.search(r"(?<=_)([a-zA-Z0-9]+)(?=/)",path).group(0)
    shutil.copy(image_path,os.path.join("image_training",classes +str(index) + ".jpg"))
    #image = cv2.imread(image_path)
    #cv2.imwrite(os.path.join("image_training",image_path + ".jpg"),image)
for index,image_path in enumerate(test_image["Path_image"]):
    classes = re.search(r"(?<=_)([a-zA-Z0-9]+)(?=/)",path).group(0)
    shutil.copy(image_path,os.path.join("image_testing",classes + str(index) +".jpg"))
    #image = cv2.imread(image_path)
    #cv2.imwrite(os.path.join("image_testing",image_path + ".jpg"),image)