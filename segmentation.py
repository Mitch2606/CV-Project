from msilib.schema import Directory
import cv2
import csv
import os
import numpy as np
import tkinter as tk
from pathlib import Path
from tkinter import filedialog as fd

#pad image based on dimensions and then resize
def resize_crop(t):
    height = t.shape[0]
    width = t.shape[1]
    if height == width:
        t = cv2.resize(t, (500,500))
        return t
    if height > width:
        delta = int((height - width)/2)
        t = cv2.copyMakeBorder(t, 0, 0, delta, delta, cv2.BORDER_REPLICATE, None, value = 0)
    else:
        delta = int((width - height)/2)
        t = cv2.copyMakeBorder(t, delta, delta, 0, 0, cv2.BORDER_REPLICATE, None, value = 0)
    t = cv2.resize(t, (500,500))
    return t

#check for smaller contours
def small_contour(c):
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	return len(approx) <= 3

#area key function
def area_box(e):
    return (e[2]*e[3])

#find if boxes are within one another
def box_check(rects):
    rects.sort(key = area_box)
    flip = rects[::-1]
    return_box = flip
    for m in rect:
        for k in flip:
            if m in return_box and m[0] > k[0] and (m[0]+m[2]) < (k[0]+k[2]) and m[1] > k[1] and (m[1]+m[3]) < (k[1] + k[3]):
                return_box.remove(m)
    return return_box


# create the file select window
root = tk.Tk()
root.title('Select image')
root.resizable(False, False)
root.geometry('300x150')
root.withdraw()

#Dataset path, clear old images before generating new ones
path = Path('itemDataset\\')
files = os.listdir(path)
for f in files:
    os.remove(str(path)+'/'+f)

#selectable filenames
img_directory = fd.askdirectory()

scale = 1
delta = 0
ddepth = cv2.CV_16S
i = 0
#load in image
for load in os.listdir(img_directory):
    image = cv2.imread(str(img_directory)+'/'+load, 1)
    image = cv2.resize(image, (1000,1000))

    #split image channels
    RBG_planes = cv2.split(image)

    #attempt to remove shadows from color channels
    result_planes = []
    result_norm_planes = []
    for plane in RBG_planes:
        dilated_img = cv2.dilate(plane, np.ones((5,5), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 9)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=100, beta=200, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    #merge and find edge of objects
    result = cv2.merge(result_planes)
    x = cv2.Sobel(result, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    y = cv2.Sobel(result, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    ax = cv2.convertScaleAbs(x)
    ay = cv2.convertScaleAbs(y)
    result = cv2.addWeighted(ax, 0.5, ay, 0.5, 0)

    #threshold detected edges and find the contours
    _, result =  cv2.threshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 30, 190, cv2.THRESH_BINARY)
    contours, hierachy = cv2.findContours(result.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #go through each contour, removing smaller contours and marking their bounding box
    rect = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 150 or small_contour(c):
            cv2.fillPoly(result, pts=[c], color=0)
            continue
        x,y,w,h = cv2.boundingRect(c)
        rect.append([x,y,w,h])
    newbox = box_check(rect)

    #save cropped images
    for box in newbox:
        cropped = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        cropped = resize_crop(cropped)
        cv2.imwrite(str(path)+"/"+str(i)+".jpg", cropped)
        i+=1

#Move to CSV
header = ["file Location","label"]
with open('SegmentedImageset.csv', 'w+', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for p in range(i):
        data = [str(p)+'.jpg', '0']
        writer.writerow(data)




