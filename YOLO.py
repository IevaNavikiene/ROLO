import os
import numpy as np
import tensorflow as tf
import cv2
import time
import sys
import pickle
sys.path.insert(0, 'utils')
import ROLO_utils as util
sys.path.insert(0, 'YOLO/')
from YOLO_network import YOLO_TF

yolo = YOLO_TF([])
test = 'Car4_2'#'Strange'#4
heatmap= False
[yolo.w_img, yolo.h_img, sequence_name, dummy_1, dummy_2]= util.choose_video_sequence(test)
root_folder = 'benchmark/DATA'
img_fold = os.path.join(root_folder, sequence_name, 'img/')
out_fold = os.path.join(root_folder, sequence_name, 'yolo_out/')
yolo.createFolder(out_fold)
yolo.prepare_training_data(img_fold,'',out_fold, without_gt=True)
