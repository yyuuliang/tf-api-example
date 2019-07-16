"""
@author: yyuuliang
project: https://github.com/yyuuliang/tf-api-example

Convert Autti's csv to gt.txt
"""

import os
import sys
import csv


def csv_txt():

    labels = {'"car"': 1,
    '"truck"': 2,
    '"pedestrian"': 3,
    '"trafficLight"': 4,
    '"biker"': 5,
    }

    csv_fname = os.path.join('dataset/autti/train.csv')
    idx = 0
    gt2 = open('dataset/autti/gt2.txt','w') 
    fidx = -1
    last_img_name = ''
    with open(csv_fname, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|', )
        for row in spamreader:
            img_name = row[0]
            if last_img_name !=  img_name:
                last_img_name = img_name
                fidx = fidx +1
            
            img_name_idx = str(fidx).zfill(5)
            xmin = int(row[1])
            ymin = int(row[2])
            xmax = int(row[3])
            ymax = int(row[4])
            class_name = row[6]
            class_id = labels[class_name]
            gtstr = '{}.jpg;{};{};{};{};{}\n'.format(img_name_idx,xmin,ymin,xmax,ymax,class_id)
            gt2.write(gtstr)
    gt2.close()


if __name__ == '__main__':
    csv_txt()