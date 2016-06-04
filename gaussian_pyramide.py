import cv2
import numpy as np,sys
import pixel as pix

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
	
def get_list(gpA, i, j):
	return [gpA[x][i][j] for x in range(6)]
	

def create_final_image():
	A = cv2.imread('../dataset/img0001-3.png')
	gpA = generate_pyramids(A)
	height, width, channels = gpA[0].shape
	final = gpA[0].copy()

for i in range(1,6):
	G = gpA[i]
	for j in range(i):
		G = cv2.pyrUp(G)
	gpA[i] = G

	for i in range(height):
		for j in range(width):
			lst = np.array(get_list(gpA, i, j))
			mean = lst.mean(axis = 0)
			final[i][j] = mean

	cv2.imwrite('../output/pyramide.png', final)
	
create_final_image()
