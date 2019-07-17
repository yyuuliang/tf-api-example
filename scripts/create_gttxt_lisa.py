"""
@author: yyuuliang
project: https://github.com/yyuuliang/tf-api-example

Convert Lisa Traffic Light csv to gt.txt
"""

import os
import sys
import csv
import cv2


def csv_txt():

    labels = {'trafficLight': 1
    }

    csv_fname = os.path.join('dataset/lisa/train.csv')
    idx = 0
    gt2 = open('dataset/lisa/gt.txt','w') 
    fidx = -1
    last_img_name = ''
    with open(csv_fname, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|', )
        # remove the first line from lisa' csv file
        for item in spamreader:
            row = item[0].split(';')
            img_name = row[0].split('--')[1]
            if last_img_name !=  img_name:
                last_img_name = img_name
                fidx = fidx +1
            img_name_idx = str(fidx).zfill(5)
            # copy the image to project folder using new name
            img_path = '/home/notus/whitebase/Dataset/LISA-traffic-lights/dayTrain/dayClip1/frames/dayClip1--'+img_name
            im = cv2.imread(img_path)
            cv2.imwrite('dataset/lisa/'+img_name_idx+'.jpg', im)
            xmin = int(row[2])
            ymin = int(row[3])
            xmax = int(row[4])
            ymax = int(row[5])
            class_id = '1'
            gtstr = '{}.jpg;{};{};{};{};{}\n'.format(img_name_idx,xmin,ymin,xmax,ymax,class_id)
            gt2.write(gtstr)
    gt2.close()


if __name__ == '__main__':
    csv_txt()
