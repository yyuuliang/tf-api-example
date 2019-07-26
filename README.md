# Tensorflow Object Detection API Example

## Introduction
The project is inspired by @aarcosg's [traffic sign detection project](https://github.com/aarcosg/traffic-sign-detection). The purpose of this project is to demenstrate how to use Tensorflow Object Detection API to train your own dataset, and use your own model to do inference.  
The main steps are:  
1. Setup the environment.  
2. Prepare the dataset.   
3. Train the models.  
4. Inference.  

## Setup

- If you have used Tensorflow Object Detection API before, then you are all set.   
- Download Tensorflow Object Detection API from [here](https://github.com/tensorflow/models), then follow the instruction [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) to finish the install.

- Download faster_rcnn_resnet50_coco pre-trained model from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and put it under this project. 
- Then modify configs/faster_rcnn_resnet50_coco_autti.config to update the weights location.  


## Dataset

The dataset used in this example is from [Autti](http://autti.co/), you can downloaded it from [here](https://github.com/udacity/self-driving-car/tree/master/annotations).  
The dataset's annotation file is a .csv file. You have to fist convert it into a gt.txt. I have write a script to do this.    

```bash
python scripts/create_gttxt.py
```
More scripts are provied in scripts folder. Currently supported dataset:  
- Autti 
- LISA traffic light dataset
- Tsinghua-Tencent 100K traffic sign dataset

Tensorflow Object Detection API use tf record data in order to train the model. TF record is a combination of annotation and images files. The scripts/create_tfrecord.py will do the work. Please modify the setting in this file according to your own setting. After that, you will get train.record and val.record at your dataset folder. Update the configs/faster_rcnn_resnet50_coco_autti.config again to use the new location.  

```bash
python scripts/create_tfrecord.py --data_dir dataset/autti --output_dir dataset/autti --label_map_path configs/autti_label_map.pbtxt
```

Pay attention to these settings in create_tfrecord.py.

```python
  num_total = 12
  num_train = 11
  num_classes = 5
  ```

## How to train

And you have everything to train the network. The first step is to use TF API's script to train the model. Modify the train.py's location to your owns.   

```bash
python ~/Github/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path configs/faster_rcnn_resnet50_coco_autti.config --train_dir dataset/autti/ckpt/
```

After that, you will find checkpoint files in ckpt folder. The last step is to export the checkpoint to useable weights files. Please notice the ckpt index should be correct.  

```bash
python ~/Github/models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet50_coco_autti.config --trained_checkpoint_prefix dataset/autti/ckpt/model.ckpt-1905 --output_directory dataset/autti/exported
```

## How to use
I have put everyhing about the inference in ref.py file. Please modify the locations in the script first, then just run it.   

```bash
python ref.py
``` 

The tensorboard log is enabled by default. Just use the following cmd to run the tensorboard.    

```bash
tensorboard  --logdir=tf-log
tensorboard --logdir=dataset/ckpt
```
Congratulations, you have finished all the work!  

## Trouble shooting

First thing first, read the instruction again [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

When run create_tfrerocd.py, error ImportError: cannot import name 'string_int_label_map_pb2'  
Make sure you have finish protobuf-compiler installation.

When run create_tfrerocd.py, error ImportError: No module named 'object_detection'  
Make sure you have config the PYTHONPATH.

## Acknowledgement
The project is inspired by @aarcosg's [traffic sign detection project](https://github.com/aarcosg/traffic-sign-detection),  some code in ref.py and scripts/create_tfrecord.py are directly from their work.   Thanks!  