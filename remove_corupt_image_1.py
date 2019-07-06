import cv2
import os
from glob import glob
import sys

paths = ['/home/s1885778/nrl/dataset/Images_/Images_5/54/']

log_file = open('log_file.txt', 'w')

for root_dir in paths:
    paths = [y for x in os.walk(root_dir) for y in glob(os.path.join(x[0], "*.jpg"))]

    for img_path in paths:
        img_id = int(img_path.split("/")[-1].split(".")[0])
        image = cv2.imread(img_path)

        if image is None:
            #print(img_path)
            os.remove(img_path)
            print("File " + img_path + " removed!")
            log_file.write(img_path)