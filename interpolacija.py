import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread('zaba.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
fig,ax= plt.subplots(figsize=(8,5))
ax.imshow(img)
plt.show()