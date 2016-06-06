import cv2
import numpy as np,sys

A = cv2.imread('../dataset/img0001-3.png')

#generate Gaussian pyramide for A
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
    
height, width, channels = gpA[0].shape
final = gpA[0].copy()
print final

for i in range(height):
    for j in range(width):
        lst = np.array([gpA[x][i][j] for x in range(6)])
        mean = lst.mean(axis = 0)
        print final[i][j]
        print mean
        final[i][j] = mean

cv2.imwrite('/pyramide.png', final)