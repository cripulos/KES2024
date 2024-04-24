import numpy as np

import cv2
import tensorflow as tf
from tensorflow.lite.python import interpreter as tflite_interpreter
import tensorflow.keras.backend as K


from scipy.spatial.distance import directed_hausdorff
import os
from utils import *
import pandas as pd
import time

# Limit size occupied by the segmentation model (tensorflow-based)
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs, ", len(logical_gpus), " Logical GPUs")
    except RuntimeError as e:
        print(e)

# Get reproducible results
tf.random.set_seed(
    42
)

img_path = "/Img/"
mask_path = "/Img/masks/"
model_path = " ../Model/"
path_test = ""
global_segmentation_model_name = "quantized_model_2.tflite"
img_shape = [512, 512]

model=model_path+global_segmentation_model_name
print("********: "+ model)
segmentator =  tflite_interpreter.Interpreter("../Model/quantized_model_2.tflite")

segmentator.allocate_tensors()
# Get input and output details
input_details = segmentator.get_input_details()
output_details = segmentator.get_output_details()

def prepare_test_data():
    """Load slides' annotations and tissue fragment images,
    before preparing masks depending on the learning type received."""
    testX = []
    test_gt = []

    img_list = os.listdir(img_path)
    for el in img_list:
        if el=="Thumbs.db":
            continue
        # Load input image to predict
        input_img = cv2.imread(img_path + el, cv2.IMREAD_COLOR)
        input_img = cv2.resize(input_img, (img_shape[0], img_shape[1]))
        input_img = cv2.UMat(input_img)

        # Load its corresponding ground truth
        gt = cv2.imread(mask_path + el.split(".")[0] + "_gt.png", cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (img_shape[0], img_shape[1]))
        gt[gt <= 15] = 0
        gt[gt > 15] = 1
        gt = np.reshape(gt, (gt.shape[0], gt.shape[1], 1))
        
        # Save in lists
        testX.append(input_img)
        test_gt.append(gt)

    return testX, test_gt

def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou = float(np.sum(intersection)) / float(np.sum(union))

    return iou

def dice_coef(y_true, y_pred, smooth=1):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


img_shape = [512, 512]
testX, test_gt = prepare_test_data()

start_time = time.time()
test_pred = []
for i, img in enumerate(testX):
    pred = predict_mask(img, segmentator, img_shape=[512, 512])
    test_pred.append(pred)
test_pred = np.asarray(test_pred)
end_time = time.time()
inference_time = end_time - start_time

print("Inference Time:", inference_time, "seconds")

test_gt = tf.convert_to_tensor(test_gt, dtype=tf.float32)
test_pred = tf.convert_to_tensor(test_pred, dtype=tf.float32)

print("test gt shape ", test_gt.shape)
print("test pred shape ", test_pred.shape)
dice = dice_coef(test_gt, test_pred)
print("dice: ", dice)

iou = calculate_iou(test_pred, test_gt)
print("==== iou: ", iou)
