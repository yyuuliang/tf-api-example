import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import tensorflow as tf

def getsize(x1,y1,x2,y2):
    w = x2-x1
    h = y2-y1
    return w,h


def draw_bb():
    impath = 'dataset/lisa/mytraining/02068.jpg'
    im = np.array(Image.open(impath), dtype=np.uint8)
    # Create figure and axes
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)
    w,h = getsize(818,398,882,462)
    # Create a Rectangle patch
    rect = patches.Rectangle((818,398),w,h,linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()


# read tfrecord

tf.enable_eager_execution()

file_path = 'dataset/lisa/mytfrecord/train.record'
filenames = [file_path]
raw_dataset = tf.data.TFRecordDataset(filenames)
print('raw: ')
print(raw_dataset)
idx = 0
# for raw_record in raw_dataset.take(10):
#     idx = idx +1
#     if idx < 10:
#         print(repr(raw_record))

# Create a description of the features.
feature_description = {
    'image/filename': tf.FixedLenFeature([], tf.string, default_value=''),
    'image/source_id': tf.FixedLenFeature([], tf.string, default_value=''),
    'image/object/class/text': tf.FixedLenFeature([], tf.string, default_value=''),
    'image/object/class/label': tf.FixedLenFeature([], tf.int64, default_value=0),
    'image/object/bbox/xmin': tf.FixedLenFeature([], tf.float32, default_value=0),
    'image/object/bbox/ymin': tf.FixedLenFeature([], tf.float32, default_value=0),
    'image/object/bbox/xmax': tf.FixedLenFeature([], tf.float32, default_value=0),
    'image/object/bbox/ymax': tf.FixedLenFeature([], tf.float32, default_value=0),
}

def _parse_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)
print('parsed: ')
print(parsed_dataset)

for parsed_record in parsed_dataset.take(10):
    idx = idx +1
    if idx < 10:
        print(repr(parsed_record))