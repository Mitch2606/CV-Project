from PIL import Image
import cv2
import requests
from io import BytesIO

import numpy as np

def resizeWithpaddingUsingImage(im, desired_size):
    old_size = im.size  # old_size[0] is in (width, height) format
    print(old_size)

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image

    # thumbnail is a in-place operation

    # im.thumbnail(new_size, Image.ANTIALIAS)

    im = im.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))

    return new_im

import csv
fileName = "nonSegmentedImageset.csv"

saveFolder = "cityDataset/"

fields = ['file Location', 'label']
rows = [];
with open("city/ec1m_distractor_medium_urls.txt", 'r') as f:
    num = 0;

    while(num != 1000):
        try:
            line = f.readline()

            response = requests.get(line)
            img = Image.open(BytesIO(response.content))
            currentFileDir = str(num) + ".jpg";

            img = resizeWithpaddingUsingImage(img, 500)
            #img.show()

            img.save(saveFolder + currentFileDir)
            num += 1;

            rows.append([currentFileDir, 0])

        except:
            print("Link broken")

    f.close()


with open(fileName, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(fields)
    csvwriter.writerows(rows)








