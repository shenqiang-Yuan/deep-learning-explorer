import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

#cell:
from tinyenv.flags import flags
_FLAGS = flags()
# cell_end.
# Call this first to load the parameters.if you don't used tinymind service,then delete this cell

sys.path.insert(0, '../libraries')
from mrcnn.config import Config
import mrcnn.utils as utils
import mrcnn.model as modellib
import mrcnn.visualize as visualize
from mrcnn.model import log
import mcoco.coco as coco
import mextra.utils as extra_utils


tf.app.flags.DEFINE_string(
    'HOME_DIR',
    _FLAGS.HOME_DIR,
    'the home directory,default value is "master"')
tf.app.flags.DEFINE_string(
    'DATA_DIR',
    _FLAGS.DATA_DIR,
    'the data directory,default value is "master/data/shapes"')
tf.app.flags.DEFINE_string(
    'TRAINED_WEIGHTS_DIR',
    'master/data/trained_model',
    'the trained wights data directory,default value is "master/data/trained_model"')
tf.app.flags.DEFINE_string(
    'SAVING_MODEL_DIR',
    'master/data/logs',
    'where the model data is to save,default value is "master/data/logs"')

tf.app.flags.DEFINE_string(
    'COCO_MODEL_PATH',
    'master/data/trained_model/mask_rcnn_coco.h5',
    'the data directory,default value is "master/data/trained_model/mask_rcnn_coco.h5"')
tf.app.flags.DEFINE_string(
    'inititalize_weights_with',
    'coco',
    'which dataset is used to acquire the initialize wights,default value is "coco"')
tf.app.flags.DEFINE_integer(
    'num_classes', _FLAGS.num_class, 'Number of classes to use in the dataset.')

tf.app.flags.DEFINE_float(
    'learning_rate', _FLAGS.learning_rate,
    'The learning rate used by a polynomial decay learning rate.')
# HOME_DIR = 'master'
# DATA_DIR = os.path.join(HOME_DIR, "data/shapes")
# WEIGHTS_DIR = os.path.join(HOME_DIR, "data/weights")
# MODEL_DIR = os.path.join(DATA_DIR, "logs")

# # Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(WEIGHTS_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
  
# dataset_train = coco.CocoDataset()
# dataset_train.load_coco(DATA_DIR, subset="shapes_train", year="2018")
# dataset_train.prepare()

# dataset_validate = coco.CocoDataset()
# dataset_validate.load_coco(DATA_DIR, subset="shapes_validate", year="2018")
# dataset_validate.prepare()

# dataset_test = coco.CocoDataset()
# dataset_test.load_coco(DATA_DIR, subset="shapes_test", year="2018")
# dataset_test.prepare()

# # Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# image_size = 64
# rpn_anchor_template = (1, 2, 4, 8, 16) # anchor sizes in pixels
# rpn_anchor_scales = tuple(i * (image_size // 16) for i in rpn_anchor_template)

class ShapesConfig(Config):
    """Configuration for training on the shapes dataset.
    """
    NAME = "shapes"

    # Train on 1 GPU and 2 images per GPU. Put multiple images on each
    # GPU if the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes (triangles, circles, and squares)

    # Use smaller images for faster training. 
    IMAGE_MAX_DIM = image_size
    IMAGE_MIN_DIM = image_size
    
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = rpn_anchor_scales

    # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 400

    VALIDATION_STEPS = STEPS_PER_EPOCH / 20
    
    def parse_and_config(self,FLAGS):
        self.NAME = FLAGS.DATA_DIR.split('/')[-1]
        self.NUM_CLASSES=FLAGS.num_classes
        self.LEARNING_RATE = FLAGS.learning_rate
    
# config = ShapesConfig()
#config.display()

# model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
# inititalize_weights_with = "coco"  # imagenet, coco, or last

# if inititalize_weights_with == "imagenet":
#     model.load_weights(model.get_imagenet_weights(), by_name=True)    
# elif inititalize_weights_with == "coco":
#     model.load_weights(COCO_MODEL_PATH, by_name=True,
#                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
#                                 "mrcnn_bbox", "mrcnn_mask"])  
# elif inititalize_weights_with == "last":
#     # Load the last model you trained and continue training
#     model.load_weights(model.find_last()[1], by_name=True)
    
# print("start training!")
# model.train((dataset_traindataset , dataset_validate, 
#             learning_rate=config.LEARNING_RATE, 
#             epochs=2,
#             layers='heads')
#fine tune
'''
model.train((dataset_traindataset , dataset_validate, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=3, # starts from the previous epoch, so only 1 additional is trained 
            layers="all")
'''

def main(_):
    FLAGS = tf.app.flags.FLAGS
    if not FLAGS.DATA_DIR:
        raise ValueError('You must supply the dataset directory with --DATA_DIR')
    COCO_MODEL_PATH = os.path.join(FLAGS.TRAINED_WEIGHTS_DIR, "mask_rcnn_coco.h5")
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(FLAGS.COCO_MODEL_PATH)
    
    DATA_DIR = FLAGS.DATA_DIR
    dataset_train = coco.CocoDataset()
    dataset_train.load_coco(DATA_DIR, subset="shapes_train", year="2018")
    dataset_train.prepare()

    dataset_validate = coco.CocoDataset()
    dataset_validate.load_coco(DATA_DIR, subset="shapes_validate", year="2018")
    dataset_validate.prepare()

    dataset_test = coco.CocoDataset()
    dataset_test.load_coco(DATA_DIR, subset="shapes_test", year="2018")
    dataset_test.prepare()
    
#     image_ids = np.random.choice(dataset_train.image_ids, 4)
#     for image_id in image_ids:
#         image = dataset_train.load_image(image_id)
#         mask, class_ids = dataset_train.load_mask(image_id)
#         visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

#     image_size = 64
#     rpn_anchor_template = (1, 2, 4, 8, 16) # anchor sizes in pixels
#     rpn_anchor_scales = tuple(i * (image_size // 16) for i in rpn_anchor_template)
    
    config = ShapesConfig()
    config.parse_and_config(FLAGS)
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=FLAGS.SAVING_MODEL_DIR)
    
    inititalize_weights_with = FLAGS.inititalize_weights_with  # imagenet, coco, or last

    if inititalize_weights_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)    
    elif inititalize_weights_with == "coco":
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])  
    elif inititalize_weights_with == "last":
    # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
    model.train(dataset_train , dataset_validate, 
            learning_rate=config.LEARNING_RATE, 
            epochs=2,
            layers='heads')
                

if __name__=='__main__':
    tf.app.run()
