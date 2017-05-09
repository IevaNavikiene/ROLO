# Copyright (c) <2016> <GUANGHAN NING>. All Rights Reserved.
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 

'''
Script File: ROLO_network_test_single.py

Description:

    ROLO is short for Recurrent YOLO, aimed at simultaneous object detection and tracking
    Paper: http://arxiv.org/abs/1607.05781
    Author: Guanghan Ning
    Webpage: http://guanghan.info/
'''

# Imports
import sys,os
sys.path.insert(0, 'utils')
import ROLO_utils as utils
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import rnn
import cv2

import numpy as np
import os.path
import time
import random


class ROLO_TF:
    disp_console = True
    restore_weights = True#False

    # YOLO parameters
    fromfile = None
    tofile_img = 'test/output.jpg'
    tofile_txt = 'test/output.txt'
    imshow = True
    filewrite_img = False
    filewrite_txt = False
    disp_console = True
    yolo_weights_file = 'weights/YOLO_small.ckpt'
    alpha = 0.1
    threshold = 0.2
    iou_threshold = 0.5
    num_class = 20
    num_box = 2
    grid_size = 7
    classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    test_folder= 'Couple2'#8
    w_img, h_img = [352, 240]

    # ROLO Network Parameters
    rolo_weights_file = '/u03/Guanghan/dev/ROLO-dev/model_dropout_20.ckpt'
    # rolo_weights_file = '/u03/Guanghan/dev/ROLO-dev/model_dropout_30.ckpt'
    lstm_depth = 3
    num_steps = 3  # number of frames as an input sequence
    num_feat = 4096
    num_predict = 6 # final output of LSTM 6 loc parameters
    num_gt = 4
    num_input = num_feat + num_predict # data input: 4096+6= 5002

    # ROLO Parameters
    batch_size = 1
    display_step = 1

    # tf Graph input
    x = tf.placeholder("float32", [None, num_steps, num_input])
    istate = tf.placeholder("float32", [None, 2*num_input]) #state & cell => 2x num_input
    y = tf.placeholder("float32", [None, num_gt])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_input, num_predict]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_predict]))
    }


    def __init__(self,argvs = []):
        print("ROLO init")
        self.ROLO(argvs)

    def LSTM_single(self, name,  _X, _istate, _weights, _biases):

        # input shape: (batch_size, n_steps, n_input)
        _X = tf.transpose(_X, [1, 0, 2])  # permute num_steps and batch_size
        # Reshape to prepare input to hidden activation
        _X = tf.reshape(_X, [self.num_steps * self.batch_size, self.num_input]) # (num_steps*batch_size, num_input)
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        if(tf.__version__ >= '1.0.1'):
            _X = tf.split(_X, self.num_steps, 0) # n_steps * (batch_size, num_input)
        else:
            _X = tf.split(0, self.num_steps, _X)
        if(tf.__version__ >= '1.0.1'):
            cell = rnn.BasicLSTMCell(self.num_input, self.num_input)#tf.nn.rnn_cell.LSTMCell(self.num_input, self.num_input)            
            initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            #with tf.variable_scope(name):
            states = [initial_state]
            outputs = []
            for step in range(self.num_steps):
               if step > 0:
                  tf.get_variable_scope().reuse_variables()
               output, new_state = cell(_X[step], states[-1])
               outputs.append(output)
               states.append(new_state)
        else:
            cell = tf.nn.rnn_cell.LSTMCell(self.num_input, self.num_input)      
            #state = [_istate]
            state = cell.zero_state(self.batch_size, dtype=tf.float32)
            for step in range(self.num_steps):
                outputs, state =  tf.nn.rnn(cell, [_X[step]], state)
                tf.get_variable_scope().reuse_variables()
        return outputs


        # Experiment with dropout
    def dropout_features(self, feature, prob):
        num_drop = int(prob * 4096)
        drop_index = random.sample(xrange(4096), num_drop)
        for i in range(len(drop_index)):
            index = drop_index[i]
            feature[index] = 0
        return feature
    '''---------------------------------------------------------------------------------------'''
    def build_networks(self):
        if self.disp_console : print "Building ROLO graph..."

        # Build rolo layers
        self.lstm_module = self.LSTM_single('lstm_test', self.x, self.istate, self.weights, self.biases)
        self.ious= tf.Variable(tf.zeros([self.batch_size]), name="ious")
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        if self.disp_console : print "Loading complete!" + '\n'


    def testing(self, x_path):
        total_loss = 0

        print("TESTING ROLO...")
        # Use rolo_input for LSTM training
        pred = self.LSTM_single('lstm_train', self.x, self.istate, self.weights, self.biases)
        print("pred: ", pred)
        self.pred_location = pred[0][:, 4097:4101]
        print("pred_location: ", self.pred_location)
        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:

            if (self.restore_weights == True):
                sess.run(init)
                self.saver.restore(sess, self.rolo_weights_file)
                print "Loading complete!" + '\n'
            else:
                sess.run(init)

            id = 0 #don't change this

            # Keep training until reach max iterations
            while id < self.testing_iters - self.num_steps:
                # Load training data & ground truth
                try:
                    batch_xs = self.rolo_utils.load_yolo_output_test(x_path, self.batch_size, self.num_steps, id) # [num_of_examples, num_input] (depth == 1)
                except ValueError:
                    print("Oops! There is something wrong with your YOLO weights")
                    id += 1
                    continue
                
                # Reshape data to get 3 seq of 5002 elements
                batch_xs = np.reshape(batch_xs, [self.batch_size, self.num_steps, self.num_input])
                pred_location= sess.run(self.pred_location,feed_dict={self.x: batch_xs, self.istate: np.zeros((self.batch_size, 2*self.num_input))})
                # Save pred_location to file
                utils.save_rolo_output_test(self.output_path, pred_location, id, self.num_steps, self.batch_size)
                id += 1
                print(id)

            print "Testing Finished!"
        return None

    def ROLO(self, argvs):
            self.test_folder = argvs['test_folder']
            self.rolo_utils= utils.ROLO_utils()
            self.rolo_utils.loadCfg()
            self.params = self.rolo_utils.params

            arguments = self.rolo_utils.argv_parser(argvs)

            if self.rolo_utils.flag_train is True:
                self.training(utils.x_path, utils.y_path)
            elif self.rolo_utils.flag_track is True:
                self.build_networks()
                self.track_from_file(utils.file_in_path)
            elif self.rolo_utils.flag_detect is True:
                self.build_networks()
                self.detect_from_file(utils.file_in_path)
            else:
                print "Default: running ROLO test."
                self.build_networks()

                [self.w_img, self.h_img, sequence_name, dummy_1, self.testing_iters] = utils.choose_video_sequence(self.test_folder)

                x_path = os.path.join('benchmark/DATA', sequence_name, 'yolo_out/')
                self.output_path = os.path.join('benchmark/DATA', sequence_name, 'rolo_out_test/')
                utils.createFolder(self.output_path)
                self.rolo_weights_file= '/home/ieva/projects/ROLO/experiments/testing/model_step3_exp3.ckpt' 
                self.testing(x_path)

    '''----------------------------------------main-----------------------------------------------------'''
def main(argvs):
        ROLO_TF(argvs)

if __name__=='__main__':
        main(' ')

