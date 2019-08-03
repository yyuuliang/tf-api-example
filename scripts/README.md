# Dataset for Autonomous Driving

### Data Augumentation
I am using this tool for data augumentation. [imgaug](https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html)  

### Autti 
Autti dataset contains vehicles on the road as long as traffic lights annotations. [Download](https://github.com/udacity/self-driving-car/tree/master/annotations).  

### LISA traffic light dataset
The dataset consists of the original LISA Traffic Light training and test data. [Downlaod](http://cvrr.ucsd.edu/vivachallenge/index.php/traffic-light/traffic-light-detection/)

### LISA traffic sign dataset
This might be the only dataset you can find if you want to train a network to do US traffic sign recognition. [Download](http://cvrr.ucsd.edu/vivachallenge/index.php/signs/sign-detection/)  
I have put a class config file for LISA's traffic sign dataset in configs folder. You can use it to create tfrecord.

### Tsinghua-Tencent 100K traffic sign dataset
The dataset provides 100000 images containing 30000 traffic-sign instances.  [Download](https://cg.cs.tsinghua.edu.cn/traffic-sign/)


# Tools
Check the tools.py in this folder for more info. 

### View and Check TFRecord file
If you have a bad tfrerord file, it will cause problems. Things like training loss is zero, or loss is very large may caused by a wrong tfrecord. In order to avoid these, I create a script to view and check tfrecord data.  

### Draw bounding box on image
This is a simple function to draw bounding box based on the xmin, ymin, xmax and ymax.  