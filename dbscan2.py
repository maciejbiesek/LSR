from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pixel as pix
import ntpath
import glob
from scipy.stats import mode
import utils
from sklearn.cluster import DBSCAN
from scipy import sparse

from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


window_size = 3

        
def get_submatrix(matrix, x, y):
    pixel = pix.Pixel(matrix[x][y])
    k = window_size
    height, width, channels = matrix.shape
    
    start_x = x - (k - 1) / 2.0 if (x - (k - 1) / 2) > 0 else 0
    stop_x = x + (k - 1) / 2.0 + 1 if (x + (k - 1) / 2) < height else height
    start_y = y - (k - 1) / 2.0 if (y - (k - 1) / 2) > 0 else 0
    stop_y = y + (k - 1) / 2.0 + 1 if (y + (k - 1) / 2) < width else width
    
    #submatrix = [[pix.Neigh() for columns in range(stop_y - start_y)] for rows in range(stop_x - start_x)]
    #submatrix = np.zeros((stop_x - start_x , stop_y - start_y))

    for i in range(int(start_x), int(stop_x)):
        for j in range(int(start_y), int(stop_y)):
            if (i != x or j != y):
                neigh = pix.Neigh(matrix[i][j])
                neigh.get_dist(pixel.color)
                pixel.add_neighbour(neigh)
            
            #submatrix[i - start_x][j - start_y].color = matrix[i][j]
            #submatrix[i - start_x][j - start_y].get_dist(matrix[x][y])
    
    pixel.sort()
    return pixel
    #return submatrix

def get_closest(not_sim_list, mode):
    closest_list = [elem for elem in not_sim_list if abs(elem.distance - mode) < 2]
    
    return closest_list

def denoise(indices, img):
    x, y = indices
    change_color(img, x, y)
                 
def change_color(img, x, y):
    sub = get_submatrix(img, x, y)
                
    not_similiar = [elem for elem in sub.neighbourhood if elem.distance > 2]
                
    distances = [elem.distance for elem in not_similiar]
    dominant = mode(distances)

    closest = get_closest(not_similiar, dominant[0][0])
                
    np_similiar = np.array([item.color for item in closest])
    color = np_similiar.mean(axis = 0)
    sub.color = color
    img[x][y] = color
    
def reshape(img):
    image_list = img.reshape((img.shape[0] * img.shape[1], 3))
    
    print "Reshaping"
    
    ind_dict = {}
    for i, colors in enumerate(img):
        for j, color in enumerate(colors):
            print i, j
            if repr(color) in ind_dict:
                ind_dict[repr(color)].append((i, j))
            else:
                ind_dict[repr(color)] = [(i, j)]
    
    return image_list, ind_dict
    
                
files = sorted(glob.glob("../dataset/*.png"))

for i in range(1):
    file_name = ntpath.basename(files[i])

    img = cv2.imread(files[i])
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_image = lab_image[:1000, :1000]
    
    print lab_image
    
    image_list, indices = reshape(lab_image)
    
    print "Compute DBSCAN"
    db = DBSCAN(eps=0.3, min_samples=10).fit(image_list)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    noised_pixels = image_list[labels == -1]
    
    print "Getting indexes of noised pixels"
    lst = []
    for pixel in noised_pixels:
        idxs = indices[repr(pixel)]
        for idx in idxs:
            if idx not in lst:
                lst.append(idx)
    
    lst.sort(key=lambda tup: (tup[0],tup[1]))
    counter = 1
    for item in lst:
        print str(counter) + "/" + str(len(lst)) + ": " + str(item)
        denoise(item, lab_image)
        counter += 1
           
            
    fin = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    path = "../output/dbscan/" + str(window_size) + "x" + str(window_size) + "/" + "faster" + file_name
    cv2.imwrite(path, fin)  