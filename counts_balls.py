# import packages
import numpy as np
import cv2

# load image
img = cv2.imread("balls.jpg")

# blur and convert to grey scale
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(grey, (11, 11), 0)

# canny edge detector
edges = cv2.Canny(blur, 30, 150)

# locate edges
(cnts, _) = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

# draw edges - drawing on img will over write the original
result = img.copy()
cv2.drawContours(result, cnts, -1, (0, 255, 0), 2)

# count the number of balls
print(len(cnts))

# write how many balls there are
cv2.putText(result, str(len(cnts)) + " balls", (0, 70), cv2.FONT_HERSHEY_DUPLEX,
            1.5, (255, 0, 0))

# show everything
cv2.imshow("original", img)
cv2.imshow("blur", blur)
cv2.imshow("edges", edges)
cv2.imshow("Result", result)
cv2.waitKey(0)
