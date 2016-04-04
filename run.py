import cv2

img = cv2.imread("dataset/img0001-3.png")

cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
cv2.imshow("RGB", img)


dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
 
cv2.namedWindow("DST", cv2.WINDOW_NORMAL)


while(1):
    cv2.imshow("DST", dst)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break
    elif k==-1:  # normally -1 returned,so don't print it
        continue
    else:
        print k # else print its value