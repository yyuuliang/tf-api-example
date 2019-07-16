## TF Object-Detection API Example

### Setup

- Download TF Object-Detection API from [here](https://github.com/tensorflow/models), then copy cognitive_planning and object_detection from models/research and put in your own dictionary.   
The dict should look like this:  

```bash
- configs
- dataset
-- autti
- models  
-- research  
--- cognitive_planning  
--- object_detection  
- scripts
- pre-train-models
- test-images
```

- Download faster_rcnn_resnet50_coco pre-trained model from the [model zoo](https://github.com/tensorflow/models)


### Dataset

The dataset used in this example is from [Autti](http://autti.co/), you can downloaded the complete dataset from [here](https://github.com/udacity/self-driving-car/tree/master/annotations)


### Script

Generate tr record

```bash
#autti
python scripts/create_tfrecord.py --data_dir dataset/autti --output_dir dataset/autti

```

train

```bash
# autti
python models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path configs/faster_rcnn_resnet50_coco_autti.config --train_dir dataset/autti/ckpt/

```


export  

```bash
#autti
python models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet50_coco_autti.config --trained_checkpoint_prefix dataset/autti/ckpt/model.ckpt-1124 --output_directory dataset/autti/exported
```

tensorboard  

```bash
tensorboard  --logdir=tf-log
tensorboard --logdir=dataset/ckpt
```
