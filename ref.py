"""
@author: aarcosg
project: https://github.com/aarcosg/traffic-sign-detection
@author: yyuuliang
project: https://github.com/yyuuliang/tf-api-example
"""


import warnings
import numpy as np
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import glob as glob
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class traffic_sign_recognation:

    TEST_IMAGE_PATH = ''
    PATH_TO_CKPT = ''
    PATH_TO_LABELS = ''
    NUM_CLASSES = 5

    def init_args(self, image_path, weights_path, labels_path):
        self.PATH_TO_CKPT = weights_path
        self.TEST_IMAGE_PATH = image_path
        self.PATH_TO_LABELS = labels_path

    
    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def detect(self, saveimg=False,use_normalized_coordinates=False):

        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # restore training checkpoint
        model_dir=self.PATH_TO_CKPT+'/model.ckpt.meta'
        config = tf.ConfigProto(allow_soft_placement = True)

        # sess = tf.Session(config = config)
        detection_graph = tf.Graph()
        with tf.Session(graph=detection_graph,config = config) as sess:
            # restore the session
            saver = tf.train.import_meta_graph(model_dir)
            saver.restore(sess,tf.train.latest_checkpoint(self.PATH_TO_CKPT))
            #read image
            image = Image.open(self.TEST_IMAGE_PATH)
            (im_width, im_height) = image.size
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = self.load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            # tensorboard record
            writer = tf.summary.FileWriter("tf-log", sess.graph)
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            writer.close()

            # vote the biggest p
            ret_b = []
            ret_c = []
            ret_p = []

            for i in range(boxes.shape[1]):
                if scores[0][i] > 0.5:
                    ymin, xmin, ymax, xmax = boxes[0][i]
                    if use_normalized_coordinates:
                        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
                    else:
                        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
                    ret_b.append([left, right, top, bottom])
                    ret_c.append([classes[0][i]])
                    ret_p.append([scores[0][i]])   
            if saveimg:
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=6)
                fstr = self.TEST_IMAGE_PATH.split('/')
                retname = fstr[len(fstr)-1].split('.')[0]
                cv2.imwrite('test-images/'+retname+'_ret.jpg', image_np)

        
        return ret_b, ret_c,ret_p


if __name__ == '__main__':
    
    # the folder contains exported weights
    PATH_TO_CKPT = 'dataset/autti/exported'

    # traffic sign labels list
    PATH_TO_LABELS = 'configs/autti_label_map.pbtxt'
    # image to test
    PATH_TO_TEST_IMAGE = 'test-images/00002.jpg'

    tfc = traffic_sign_recognation()
    tfc.init_args(PATH_TO_TEST_IMAGE,PATH_TO_CKPT,PATH_TO_LABELS)
    ret_b, ret_c,ret_p = tfc.detect(True)

    # bounding box results
    print(ret_b)
    # class labels
    print(ret_c)
    # scores of predictions
    print(ret_p)