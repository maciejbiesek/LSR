from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pixel as pix
import ntpath
import glob
import fuzzy as fz
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
    closest_list = [elem for elem in not_sim_list if fz.is_near.eval(lab = abs(elem.distance - mode))]
    
    return closest_list

def denoise(pixel, img, denoised):
    for i, colors in enumerate(img):
        for j, color in enumerate(colors):
            if (color == pixel).all():
                if (i, j) not in denoised:
                    print i, j
                    denoised.append((i, j))
                    change_color(img, i, j)
                
def change_color(img, x, y):
    sub = get_submatrix(img, x, y)
                
    not_similiar = [elem for elem in sub.neighbourhood if fz.not_similiar.eval(lab = elem.distance)]
                
    distances = [elem.distance for elem in not_similiar]
    dominant = mode(distances)

    closest = get_closest(not_similiar, dominant[0][0])
                
    np_similiar = np.array([item.color for item in closest])
    color = np_similiar.mean(axis = 0)
    sub.color = color
    img[x][y] = color
    
                

files = sorted(glob.glob("../dataset/*.png"))

for i in range(1):
    file_name = ntpath.basename(files[i])

    img = cv2.imread(files[i])
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    print "Reshaping"
    #reshape the image to be a list of pixels
    image_list = lab_image.reshape((lab_image.shape[0] * lab_image.shape[1], 3))
    
    print "Compute DBSCAN"
    db = DBSCAN(eps=2, min_samples=10).fit(image_list)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print list(set(labels))
    
    noised_pixels = image_list[labels == -1]
    
    print "Denoising"
    count = 1
    denoised = []
    for pixel in noised_pixels:
        print "\n" + str(count) + "/" + str(len(noised_pixels))
        denoise(pixel, lab_image, denoised)
        count += 1
           
            
    fin = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    path = "../output/dbscan/" + str(window_size) + "x" + str(window_size) + "/" + file_name
    cv2.imwrite(path, fin)  