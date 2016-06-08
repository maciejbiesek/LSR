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

files = sorted(glob.glob("../dataset/*.png"))

for i in range(1):
    file_name = ntpath.basename(files[i])

    img = cv2.imread(files[i])
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    height, width, channels = lab_image.shape
    
    # reshape the image to be a list of pixels
    #image_list = lab_image.reshape((lab_image.shape[0] * lab_image.shape[1], 3))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(1)
    plt.axis("off")
    plt.imshow(img)
    

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    #hist = utils.centroid_histogram(clt)
    #bar = utils.plot_colors(hist, clt.cluster_centers_)
    #bar = cv2.cvtColor(bar, cv2.COLOR_LAB2RGB)
    
    # show our color bart
    #plt.figure(2)
    #plt.axis("off")
    #plt.imshow(bar)
    
    #plt.show()

    for i in range(height):
        for j in range(width):
            
            print i, j
            
            sub = get_submatrix(lab_image, i, j)
                
            not_similiar = [elem for elem in sub.neighbourhood if fz.not_similiar.eval(lab = elem.distance)]

            num = float(len(not_similiar)) / len(sub.neighbourhood)
            if (fz.is_noised.eval(neighbours = num)):
                colors = [elem.color for elem in sub.neighbourhood]
                
                clt = KMeans(n_clusters = 2)
                clt.fit(colors)
                
                hist = utils.centroid_histogram(clt)
                
                colors_labels = zip(hist, clt.cluster_centers_)
                colors_labels.sort(key = lambda x : x[0], reverse = True)

                color = colors_labels[0][1].astype("uint8")
                print color
                sub.color = color
                lab_image[i][j] = color
                
                #distances = [elem.distance for elem in not_similiar]
                #dominant = mode(distances)

                #closest = get_closest(not_similiar, dominant[0][0])
                
                #np_similiar = np.array([item.color for item in closest])
                #color = np_similiar.mean(axis = 0)
                #sub.color = color
                #lab_image[i][j] = color
            
    fin = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    path = "../output/" + file_name
    #path = "../output/normal/" + str(window_size) + "x" + str(window_size) + "/" + file_name
    cv2.imwrite(path, fin)  