import cv2
import numpy as np,sys

A=cv2.imread('dataset/img0001-3.png')

#generate Gaussian pyramide for A
G=A.copy()
gpA=[G]
for i in xrange(6):
	G=cv2.pyrDown(G)
	gpA.append(G)

for i in range(1,6):
	G=gpA[i]
	G=cv2.pyrUp(G)
	gpA.insert(i, G)

height=[]
width=[]
channels=[]

height, width, channels = gpA.shape

G=gpA[1]
for i in range(height):
	for j in range(width):
            
		print i, j
		a=(gpA[1][i][j]+gpA[2][i][j]+gpA[3][i][j]+gpA[4][i][j]+gpA[5][i][j]+gpA[6][i][j])/6
		G.insert(i, j, a)

cv2.imwrite('pyramide_all.png', G)		            
            
'''
cv2.imwrite('pyramide0.png', gpA[0])
cv2.imwrite('pyramide1.png', gpA[1])
cv2.imwrite('pyramide2.png', gpA[2])
cv2.imwrite('pyramide3.png', gpA[3])
cv2.imwrite('pyramide4.png', gpA[4])
cv2.imwrite('pyramide5.png', gpA[5])
'''