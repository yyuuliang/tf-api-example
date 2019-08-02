"""
Data Augumentation Example
"""


import imageio
import imgaug as ia
from imgaug import augmenters as iaa 


from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
# %matplotlib inline
ia.seed(1)


image = imageio.imread("test-images/aug-test/00318.jpg")

bbs = BoundingBoxesOnImage([
    BoundingBox(x1=1070,y1=330,x2=1152,y2=448,label='18'),
    BoundingBox(x1=1079,y1=254,x2=1128,y2=332,label='1')

], shape=image.shape)

ia.imshow(bbs.draw_on_image(image, size=2))

# apply augumentation
#  We choose a simple contrast augmentation (affects only the image) and an affine transformation (affects image and bounding boxes).
seq = iaa.Sequential([
    iaa.GammaContrast(1.5),
    iaa.Affine(translate_percent={"x": 0.1}, scale=0.8)
])

image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
ia.imshow(bbs_aug.draw_on_image(image_aug, size=2))

#  we apply an affine transformation consisting only of rotation.
image_aug, bbs_aug = iaa.Affine(rotate=50)(image=image, bounding_boxes=bbs)
ia.imshow(bbs_aug.draw_on_image(image_aug))