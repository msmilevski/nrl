import cv2
import os
from glob import glob
import sys

paths = ['/home/s1885778/nrl/dataset/Images_/Images_9/95/',
         '/home/s1885778/nrl/dataset/Images_/Images_8/84/',
         '/home/s1885778/nrl/dataset/Images_/Images_8/81/',
         '/home/s1885778/nrl/dataset/Images_/Images_7/74/',
         '/home/s1885778/nrl/dataset/Images_/Images_0/6/',
         '/home/s1885778/nrl/dataset/Images_/Images_6/67/',
         '/home/s1885778/nrl/dataset/Images_/Images_6/61/',
         '/home/s1885778/nrl/dataset/Images_/Images_4/41/',
         '/home/s1885778/nrl/dataset/Images_/Images_3/36/',
         '/home/s1885778/nrl/dataset/Images_/Images_1/13/',
         '/home/s1885778/nrl/dataset/Images_/Images_1/14/',
         '/home/s1885778/nrl/dataset/Images_/Images_5/54/']

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