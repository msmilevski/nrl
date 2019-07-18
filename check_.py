import cv2
import pickle

dict = pickle.load(open('dataset/Image_embed_dict.pickle', 'rb'))

general_path = "/home/s1885778/nrl/dataset/Images_/Images_"

for item in dict:
    image_paths = []
    image_ids = dict[item]
    for id in image_ids:
        temp = general_path + str(item) + "/" + str(id)
        image = cv2.imread(temp)
        print(temp)



