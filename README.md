# TF Object-Detection API Example

## Setup

- Download TF Object-Detection API from [here](https://github.com/tensorflow/models), then follow the instruction [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) to finish the install.

- Download faster_rcnn_resnet50_coco pre-trained model from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and put it under this project. Then modify configs/faster_rcnn_resnet50_coco_autti.config to update the new weights location.


## Dataset

The dataset used in this example is from [Autti](http://autti.co/), you can downloaded the complete dataset from [here](https://github.com/udacity/self-driving-car/tree/master/annotations).  
The dataset's ground truth is a .csv file. You have to fist convert it into a gt.txt. I have write a script to do this, find it at scripts/create_gttxt.py.  
The dataset will be eventually convert into tfrecord format. The python file scripts/create_tfrecord.py will do the work. Please modify the setting in this file according to your own setting. After that, you will get train.record and val.record at your dataset folder. Update the configs/faster_rcnn_resnet50_coco_autti.config again to use the new location.  

```bash
python scripts/create_tfrecord.py --data_dir dataset/autti --output_dir dataset/autti
```

## How to train

And you have everything to train the network. The first step is to use TF API's script to train the model. 

```bash
# autti
python ~/Github/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path configs/faster_rcnn_resnet50_coco_autti.config --train_dir dataset/autti/ckpt/

```
After that, you will have checkpoint files in ckpt folder, use the below script to export the checkpoint to a useable weights file. Please notice the ckpt index should be correct.  

```bash
python ~/Github/models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet50_coco_autti.config --trained_checkpoint_prefix dataset/autti/ckpt/model.ckpt-1905 --output_directory dataset/autti/exported
```

## How to use the model
I have put everyhing in ref.py file, first modify the setting in this file, then just run it. Congratulations, you have finished all the work!  

```bash
python ref.py
``` 

The tensorboard log is enabled by default. Just use the following cmd to run the tensorboard.    

```bash
tensorboard  --logdir=tf-log
tensorboard --logdir=dataset/ckpt
```


## Trouble shooting

First thing first, read the instruction again [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

When run create_tfrerocd.py, error ImportError: cannot import name 'string_int_label_map_pb2'  
Make sure you have finish protobuf-compiler installation.

When run create_tfrerocd.py, error ImportError: No module named 'object_detection'  
Make sure you have config the PYTHONPATH.
