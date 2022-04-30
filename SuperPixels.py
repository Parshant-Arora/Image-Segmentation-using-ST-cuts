import numpy as np
import math
import cv2

def is_inside(x,y,h,w):
    return 0 < x and x < h and 0 < y and y < w


def CreateSP(Img, region_size):
    I = np.array(Img, dtype=np.int64)
    # I.astype(np.int64) 
    h = I.shape[0]
    w = I.shape[1]
    N = I.shape[0] * I.shape[1]
    K = region_size
    S = K
    S = int(S)
    labels = -1 * np.ones((I.shape[0], I.shape[1]))
    d = math.inf * np.ones((I.shape[0], I.shape[1]))
    m = 2.73
    SP_cnt = 0
    SP_dict = {}
    for i in range(1, int((I.shape[0]+S//2)/S)+1):
        for j in range(1, int((I.shape[1]+S//2)/S)+1):
            x = S*i - S//2
            y = S*j - S//2
            if x < h and y < w:
                # I[x][y] = (0,0,255)
                SP_dict[SP_cnt] = (x, y)
                SP_cnt += 1
    # print(SP_dict)
    for sp in SP_dict:
        x, y = SP_dict[sp]
        print(x, y, sp)
        for i in range(x-S, x+S):
            for j in range(y-S, y+S):
                if i < h and j < w and i >= 0 and j >= 0:
                    lab1 = np.array([I[i][j][0], I[i][j][1], I[i][j][2]])
                    lab2 = np.array([I[x][y][0], I[x][y][1], I[x][y][2]])
                    de = np.linalg.norm(lab1 - lab2)
                    ds = np.linalg.norm(np.array([x, y]) - np.array([i, j]))
                    D = math.sqrt(de*de + (ds/S)*(ds/S)*m*m)
                    if D < d[i][j]:
                        d[i][j] = D
                        labels[i][j] = sp
                        # print(sp)
    SP_mean = {}
    SP_pixel_cnt = {}
    for x in range(0, h):
        for y in range(0, w):
            sp = labels[x][y]
            if sp in SP_mean:
                # print("yes ", sp ," is in Sp mean")
                # print(SP_mean[sp])
                # print(I[x][y])
                SP_mean[sp] = SP_mean[sp]+ I[x][y]
                # print(SP_mean[sp])
                SP_pixel_cnt[sp] += 1
            else:
                SP_mean[sp] = I[x][y]
                SP_pixel_cnt[sp] = 1
    for sp in SP_mean:
        # print("Sp sum,cnt ", SP_mean[sp], SP_pixel_cnt[sp])
        SP_mean[sp] = SP_mean[sp]/SP_pixel_cnt[sp]
    for x in range(0,h):
        for y in range(0,w):
            sp = labels[x][y]
            I[x][y] = SP_mean[sp]
            # if (is_inside(x+1,y,h,w) and labels[x+1][y] != labels[x][y]) or (is_inside(x-1,y,h,w) and labels[x-1][y] != labels[x][y]) or (is_inside(x,y+1,h,w) and labels[x][y+1] != labels[x][y]) or (is_inside(x,y-1,h,w) and labels[x][y-1] != labels[x][y]):
            #     I[x][y] = [128,128,128]
                
            
            
            # print("Mean", I[x][y])
    if -1 in labels:
        print("Sad")
    else:
        print("yay")    
    
    
        
    return I
    
    
Img = cv2.imread('bunny.png')    
print(Img.shape)
region_size = 30
I = CreateSP(Img, region_size)

# cv2.imshow("hi", I*255)
# cv2.waitKey(0)
# print(labels[1:10][1:10])
# print(labels.shape)
# print(labels)

# cv2.imshow("hi", labels/500)
# cv2.waitKey(0)

    
# print(labels[labels==-1])

# cv2.imshow("hi", np.array(I, dtype = np.uint8))
# cv2.waitKey(0)

cv2.imwrite('bunny' + str(region_size) + '.png', I)
