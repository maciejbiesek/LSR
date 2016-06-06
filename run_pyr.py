from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pixel as pix
import ntpath
import glob
from scipy.stats import mode
import utils

window_size = 3

#generate Gaussian pyramide for A
def generate_pyramids(A):
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpA.append(G)

    for i in range(1,6):
        G = gpA[i]
        for j in range(i):
            G = cv2.pyrUp(G)
        gpA[i] = G

    return gpA

def get_submatrix(matrix, x, y):
    pixel = pix.Pixel(matrix[x][y])
    k = window_size
    height, width, channels = matrix.shape
    
    start_x = x - (k - 1) / 2.0 if (x - (k - 1) / 2) > 0 else 0
    stop_x = x + (k - 1) / 2.0 + 1 if (x + (k - 1) / 2) < height else height
    start_y = y - (k - 1) / 2.0 if (y - (k - 1) / 2) > 0 else 0
    stop_y = y + (k - 1) / 2.0 + 1 if (y + (k - 1) / 2) < width else width

    for i in range(int(start_x), int(stop_x)):
        for j in range(int(start_y), int(stop_y)):
            if (i != x or j != y):
                neigh = pix.Neigh(matrix[i][j])
                neigh.get_dist(pixel.color)
                pixel.add_neighbour(neigh)
    
    pixel.sort()
    return pixel
    
def get_closest(not_sim_list, mode):
    closest_list = [elem for elem in not_sim_list if abs(elem.distance - mode) < 2]
    
    return closest_list


files = sorted(glob.glob("../dataset/*.png"))

for i in range(1):
    file_name = ntpath.basename(files[i])

    img = cv2.imread(files[i])
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gpA = generate_pyramids(lab_image)

    height, width, channels = lab_image.shape
    
    for i in range(height):
        for j in range(width):
            
            print i, j
            
            sub = get_submatrix(lab_image, i, j)   
            not_similiar = [elem for elem in sub.neighbourhood if elem.distance > 2]
            num = float(len(not_similiar)) / len(sub.neighbourhood)

            if (num > 0.3):
                
                lst = np.array([gpA[x][i][j] for x in range(6)])
                mean = lst.mean(axis = 0)
                sub.color = color
                lab_image[i][j] = color
            
    fin = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    path = "../output/" + "pyr_" + file_name
    cv2.imwrite(path, fin)  