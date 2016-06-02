import cv2
import numpy as np
import ntpath
import glob

files = sorted(glob.glob("../dataset/*.png"))

for i in range(5):
    file_name = ntpath.basename(files[i])

    img = cv2.imread(files[i])
    
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    
    path = "../output/auto/" + file_name
    cv2.imwrite(path, dst)