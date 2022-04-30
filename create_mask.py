from __future__ import division
import cv2
import numpy as np
import os
import sys
import argparse
from multiprocessing import Queue
from math import exp, pow



SIGMA = 30
# LAMBDA = 1
OBJCOLOR= (0, 0, 255)
OBJCODE = 1
OBJ= "OBJ"

CUTCOLOR = (0, 0, 255)

SOURCE, SINK = -2, -1
SF = 10
LOADSEEDS = False
# drawing = False

def show_image(image):
    windowname = "Segmentation"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow(windowname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def plantSeed(image):

    def drawLines(x, y, pixelType):
        if pixelType == OBJ:
            color, code = OBJCOLOR, OBJCODE
       
        cv2.circle(image, (x, y), radius, color, thickness)
        cv2.circle(seeds, (x // SF, y // SF), radius // SF, code, thickness)

    def onMouse(event, x, y, flags, pixelType):
        global drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    def paintSeeds(pixelType):
        print("Planting", pixelType, "seeds")
        global drawing
        drawing = False
        windowname = "Plant " + pixelType + " seeds"
        cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(windowname, onMouse, pixelType)
        while (1):
            cv2.imshow(windowname, image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
    
    
    seeds = np.zeros(image.shape, dtype="uint8")
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # image = cv2.resize(image, (0, 0), fx=SF, fy=SF)

    radius = 2
    thickness = -1 # fill the whole circle
    global drawing
    drawing = False
    

    paintSeeds(OBJ)
    # paintSeeds(BKG)
    return image



# Large when ip - iq < sigma, and small otherwise
def boundaryPenalty(ip, iq):
    bp = 100 * exp(- pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2)))
    return bp

def buildGraph(image):
    seededImage = plantSeed(image)
    return seededImage






def imageSegmentation(imagefile, size=(30, 30)):
    pathname = os.path.splitext(imagefile)[0]
    image = cv2.imread(imagefile)
    # image = cv2.resize(image, size)
    seededImage = buildGraph(image)

    # print("seeded image",seededImage)
    # cv2.imwrite(pathname + "seeded.jpg", seededImage)
    # cv2.imshow("hi", seededImage)
    r,c,rgb = image.shape
    print(image.shape)
    mask = np.zeros([r,c,rgb],dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            # print(seededImage[i][j])
            if(tuple(seededImage[i][j])==OBJCOLOR):
                mask[i][j] = (255,255,255)
    # cv2.waitKey(0)
    cv2.imwrite('deer_mask.png',mask)

    

def parseArgs():
    

    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile")
    parser.add_argument("--size", "-s", 
                        default=30, type=int,
                        help="Defaults to 30x30")
    # parser.add_argument("--algo", "-a", default="ap", type=algorithm)
    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    imageSegmentation(args.imagefile, (args.size, args.size))
    





