import cv2
import numpy as np

I = cv2.imread('deer.png')
mask = cv2.imread('./masks/deer_mask.png')
print(I.shape)
print(mask.shape)
# mask = mask[:][:][0]/255
mask_2d = np.zeros((I.shape[0], I.shape[1]))
for i in range(0,I.shape[0]):
    for j in range(0,I.shape[1]):
        if mask[i][j][0] == 0:
            mask_2d[i][j] = 0
            I[i][j] = [0,0,0]
        

print(mask_2d)

# Final = cv2.bitwise_and(I, I, mask=mask_2d)


cv2.imwrite("./masks/deer_actual.png",I)
