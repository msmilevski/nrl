import cv2
import os
from glob import glob
import sys

root_dir = sys.argv[1]
paths = [y for x in os.walk(root_dir) for y in glob(os.path.join(x[0], "*.jpg"))]

for img_path in paths:
    img_id = int(img_path.split("/")[-1].split(".")[0])
    image = cv2.imread(img_path)

    if image is None:
        print(img_path)

