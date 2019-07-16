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
