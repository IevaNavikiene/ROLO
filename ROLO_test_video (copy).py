'''
Script File: ROLO_test_video.py

Description:

    This script helps to test ROLO on your video (.mp4) files without any additional knowledge
'''

import cv2
import os
import numpy as np
import sys
sys.path.insert(0, 'utils')
import ROLO_utils as utils
sys.path.insert(0, 'experiments/testing/')
sys.path.insert(0, 'YOLO/')
import tensorflow as tf

#from YOLO_network import YOLO_TF

import ROLO_utils as util
from tensorflow.contrib import rnn
import cv2

import numpy as np
import os.path
import time
import random


def ROLO_TF():
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


def YOLO_TF():
    fromfile = None
    tofile_img = 'test/output.jpg'
    tofile_txt = 'test/output.txt'
    imshow = True
    filewrite_img = False
    filewrite_txt = False
    disp_console = True
    weights_file = 'weights/YOLO_small.ckpt'
    alpha = 0.1
    threshold = 0.08
    iou_threshold = 0.5
    num_class = 20
    num_box = 2
    grid_size = 7
    classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

    w_img, h_img = [352, 240]
    num_feat = 4096
    num_predict = 6 # final output of LSTM 6 loc parameters
    num_heatmap = 1024
	self.argv_parser(argvs)
	self.build_networks()
	if self.fromfile is not None: self.detect_from_file(self.fromfile)

    def argv_parser(self,argvs):
        for i in range(1,len(argvs),2):
            if argvs[i] == '-fromfile' : self.fromfile = argvs[i+1]
            if argvs[i] == '-tofile_img' : self.tofile_img = argvs[i+1] ; self.filewrite_img = True
            if argvs[i] == '-tofile_txt' : self.tofile_txt = argvs[i+1] ; self.filewrite_txt = True
            if argvs[i] == '-imshow' :
                if argvs[i+1] == '1' :self.imshow = True
                else : self.imshow = False
            if argvs[i] == '-disp_console' :
                if argvs[i+1] == '1' :self.disp_console = True
                else : self.disp_console = False
                
    def build_networks(self):
        if self.disp_console : print "Building YOLO_small graph..."
        self.x = tf.placeholder('float32',[None,448,448,3])
        self.conv_1 = self.conv_layer(1,self.x,64,7,2)
        self.pool_2 = self.pooling_layer(2,self.conv_1,2,2)
        self.conv_3 = self.conv_layer(3,self.pool_2,192,3,1)
        self.pool_4 = self.pooling_layer(4,self.conv_3,2,2)
        self.conv_5 = self.conv_layer(5,self.pool_4,128,1,1)
        self.conv_6 = self.conv_layer(6,self.conv_5,256,3,1)
        self.conv_7 = self.conv_layer(7,self.conv_6,256,1,1)
        self.conv_8 = self.conv_layer(8,self.conv_7,512,3,1)
        self.pool_9 = self.pooling_layer(9,self.conv_8,2,2)
        self.conv_10 = self.conv_layer(10,self.pool_9,256,1,1)
        self.conv_11 = self.conv_layer(11,self.conv_10,512,3,1)
        self.conv_12 = self.conv_layer(12,self.conv_11,256,1,1)
        self.conv_13 = self.conv_layer(13,self.conv_12,512,3,1)
        self.conv_14 = self.conv_layer(14,self.conv_13,256,1,1)
        self.conv_15 = self.conv_layer(15,self.conv_14,512,3,1)
        self.conv_16 = self.conv_layer(16,self.conv_15,256,1,1)
        self.conv_17 = self.conv_layer(17,self.conv_16,512,3,1)
        self.conv_18 = self.conv_layer(18,self.conv_17,512,1,1)
        self.conv_19 = self.conv_layer(19,self.conv_18,1024,3,1)
        self.pool_20 = self.pooling_layer(20,self.conv_19,2,2)
        self.conv_21 = self.conv_layer(21,self.pool_20,512,1,1)
        self.conv_22 = self.conv_layer(22,self.conv_21,1024,3,1)
        self.conv_23 = self.conv_layer(23,self.conv_22,512,1,1)
        self.conv_24 = self.conv_layer(24,self.conv_23,1024,3,1)
        self.conv_25 = self.conv_layer(25,self.conv_24,1024,3,1)
        self.conv_26 = self.conv_layer(26,self.conv_25,1024,3,2)
        self.conv_27 = self.conv_layer(27,self.conv_26,1024,3,1)
        self.conv_28 = self.conv_layer(28,self.conv_27,1024,3,1)
        self.fc_29 = self.fc_layer(29,self.conv_28,512,flat=True,linear=False)
        self.fc_30 = self.fc_layer(30,self.fc_29,4096,flat=False,linear=False)
        #skip dropout_31
        self.fc_32 = self.fc_layer(32,self.fc_30,1470,flat=False,linear=True)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        print(self.weights_file,'self.weights_file')
        self.saver.restore(self.sess,self.weights_file)
        if self.disp_console : print "Loading complete!" + '\n'

    def conv_layer(self,idx,inputs,filters,size,stride):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))

        pad_size = size//2
        pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
        inputs_pad = tf.pad(inputs,pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')    
        conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')    
        if self.disp_console : print '    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx,size,size,stride,filters,int(channels))
        return tf.maximum(self.alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')

    def pooling_layer(self,idx,inputs,size,stride):
        if self.disp_console : print '    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx,size,size,stride)
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')

    def fc_layer(self,idx,inputs,hiddens,flat = False,linear = False):
        input_shape = inputs.get_shape().as_list()        
        if flat:
            dim = input_shape[1]*input_shape[2]*input_shape[3]
            inputs_transposed = tf.transpose(inputs,(0,3,1,2))
            inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs
        weight = tf.Variable(tf.truncated_normal([dim,hiddens], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))    
        if self.disp_console : print '    Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (idx,hiddens,int(dim),int(flat),1-int(linear))    
        if linear : return tf.add(tf.matmul(inputs_processed,weight),biases,name=str(idx)+'_fc')
        ip = tf.add(tf.matmul(inputs_processed,weight),biases)
        return tf.maximum(self.alpha*ip,ip,name=str(idx)+'_fc')

    def detect_from_cvmat(self,img):
        s = time.time()
        self.h_img,self.w_img,_ = img.shape
        img_resized = cv2.resize(img, (448, 448))
        img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray( img_RGB )
        inputs = np.zeros((1,448,448,3),dtype='float32')
        inputs[0] = (img_resized_np/255.0)*2.0-1.0
        in_dict = {self.x: inputs}
        net_output = self.sess.run(self.fc_32,feed_dict=in_dict)
        self.result = self.interpret_output(net_output[0])
        self.show_results(img,self.result)
        strtime = str(time.time()-s)
        if self.disp_console : print 'Elapsed time : ' + strtime + ' secs' + '\n'

    def detect_from_file(self,filename):
        if self.disp_console : print 'Detect from ' + filename
        img = cv2.imread(filename)
        #img = misc.imread(filename)
        self.detect_from_cvmat(img)

    def detect_from_crop_sample(self):
        self.w_img = 640
        self.h_img = 420
        f = np.array(open('person_crop.txt','r').readlines(),dtype='float32')
        inputs = np.zeros((1,448,448,3),dtype='float32')
        for c in range(3):
            for y in range(448):
                for x in range(448):
                    inputs[0,y,x,c] = f[c*448*448+y*448+x]

        in_dict = {self.x: inputs}
        net_output = self.sess.run(self.fc_32,feed_dict=in_dict)
        self.boxes, self.probs = self.interpret_output(net_output[0])
        img = cv2.imread('person.jpg')
        self.show_results(self.boxes,img)

    def interpret_output(self,output):
        probs = np.zeros((7,7,2,20))
        class_probs = np.reshape(output[0:980],(7,7,20))
        scales = np.reshape(output[980:1078],(7,7,2))
        boxes = np.reshape(output[1078:],(7,7,2,4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))

        boxes[:,:,:,0] += offset
        boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
        boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
        boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
        boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])
        
        boxes[:,:,:,0] *= self.w_img
        boxes[:,:,:,1] *= self.h_img
        boxes[:,:,:,2] *= self.w_img
        boxes[:,:,:,3] *= self.h_img

        for i in range(2):
            for j in range(20):
                probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

        filter_mat_probs = np.array(probs>=self.threshold,dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]
        
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0 : continue
            for j in range(i+1,len(boxes_filtered)):
                if self.iou(boxes_filtered[i],boxes_filtered[j]) > self.iou_threshold : 
                    probs_filtered[j] = 0.0
        
        filter_iou = np.array(probs_filtered>0.0,dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

        return result

    def show_results(self,img,results):
        img_cp = img.copy()
        if self.filewrite_txt :
            ftxt = open(self.tofile_txt,'w')
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3])//2
            h = int(results[i][4])//2
            if self.disp_console : print '    class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(y) + ',' + str(int(results[i][3])) + ',' + str(int(results[i][4]))+'], Confidence = ' + str(results[i][5])
            if self.filewrite_img or self.imshow:
                cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
                cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            if self.filewrite_txt :                
                ftxt.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h)+',' + str(results[i][5]) + '\n')
        if self.filewrite_img : 
            if self.disp_console : print '    image file writed : ' + self.tofile_img
            cv2.imwrite(self.tofile_img,img_cp)            
        if self.imshow :
            cv2.imshow('YOLO_small detection',img_cp)
            cv2.waitKey(0)
        if self.filewrite_txt : 
            if self.disp_console : print '    txt file writed : ' + self.tofile_txt
            ftxt.close()

    def iou(self,box1,box2):
        tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
        lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

    # my addition
    def createFolder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def debug_location(self, img, location):
        img_cp = img.copy()
        x = int(location[1])
        y = int(location[2])
        w = int(location[3])//2
        h = int(location[4])//2
        cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
        cv2.putText(img_cp, str(location[0]) + ' : %.2f' % location[5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
        cv2.imshow('YOLO_small detection',img_cp)
        cv2.waitKey(1)

    def debug_locations(self, img, locations):
        img_cp = img.copy()
        for location in locations:
            x = int(location[1])
            y = int(location[2])
            w = int(location[3])//2
            h = int(location[4])//2
            cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
            cv2.putText(img_cp, str(location[0]) + ' : %.2f' % location[5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
        cv2.imshow('YOLO_small detection',img_cp)
        cv2.waitKey(1)

    def debug_gt_location(self, img, location):
        img_cp = img.copy()
        x = int(location[0])
        y = int(location[1])
        w = int(location[2])
        h = int(location[3])
        cv2.rectangle(img_cp,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow('gt',img_cp)
        cv2.waitKey(1)

    def file_to_img(self, filepath):
        img = cv2.imread(filepath)
        return img


    def file_to_video(self, filepath):
        try:
                video = cv2.VideoCapture(filepath)
        except IOError:
                print 'cannot open video file: ' + filepath
        else:
                print 'unknown error reading video file'
        return video

    def iou(self,box1,box2):
            tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
            lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
            if tb < 0 or lr < 0 : intersection = 0
            else : intersection =  tb*lr
            return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)


    def find_iou_cost(self, pred_locs, gts):
            # for each element in the batch, find its iou. output a list of ious.
            cost = 0
            batch_size= len(pred_locs)
            assert (len(gts)== batch_size)
            print("batch_size: ")
            ious = []
            for i in range(batch_size):
                    pred_loc = pred_locs[i]
                    gt = gts[i]
                    iou_ = self.iou(pred_loc, gt)
                    ious.append(self, iou_)
            return ious


    def load_folder(self, path):
            paths = [os.path.join(path,fn) for fn in next(os.walk(path))[2]]
            #return paths
            return sorted(paths)


    def load_dataset_gt(self, gt_file):
            txtfile = open(gt_file, "r")
            lines = txtfile.read().split('\n')  #'\r\n'
            return lines


    def find_gt_location(self, lines, id):
            line = lines[id]
            elems = line.split('\t')   # for gt type 2
            if len(elems) < 4:
                elems = line.split(',') #for gt type 1
            x1 = elems[0]
            y1 = elems[1]
            w = elems[2]
            h = elems[3]
            gt_location = [int(x1), int(y1), int(w), int(h)]
            return gt_location


    def find_best_location(self, locations, gt_location):
            # locations (class, x, y, w, h, prob); (x, y) is the middle pt of the rect
            # gt_location (x1, y1, w, h)
            x1 = gt_location[0]
            y1 = gt_location[1]
            w = gt_location[2]
            h = gt_location[3]
            gt_location_revised= [x1 + w/2, y1 + h/2, w, h]

            max_ious= 0
            for id, location in enumerate(locations):
                    location_revised = location[1:5]
                    print("location: ", location_revised)
                    print("gt_location: ", gt_location_revised)
                    ious = self.iou(location_revised, gt_location_revised)
                    if ious >= max_ious:
                            max_ious = ious
                            index = id
            print("Max IOU: " + str(max_ious))
            if max_ious != 0:
                best_location = locations[index]
                class_index = self.classes.index(best_location[0])
                best_location[0]= class_index
                return best_location
            else:   # it means the detection failed, no intersection with the ground truth
                return [0, 0, 0, 0, 0, 0]


    def save_yolo_output(self, out_fold, yolo_output, filename):
        name_no_ext= os.path.splitext(filename)[0]
        output_name= name_no_ext
        path = os.path.join(out_fold, output_name)
        np.save(path, yolo_output)


    def location_from_0_to_1(self, wid, ht, location):
        location[1] /= wid
        location[2] /= ht
        location[3] /= wid
        location[4] /= ht
        return location

    def gt_location_from_0_to_1(self, wid, ht, location):
        wid *= 1.0
        ht *= 1.0
        location[0] /= wid
        location[1] /= ht
        location[2] /= wid
        location[3] /= ht
        return location

    def locations_normal(self, wid, ht, locations):
        wid *= 1.0
        ht *= 1.0
        locations[1] *= wid
        locations[2] *= ht
        locations[3] *= wid
        locations[4] *= ht
        return locations

    def cal_yolo_loss(self, location, gt_location):
        # Translate yolo's box mid-point (x0, y0) to top-left point (x1, y1), in order to compare with gt
        location[0] = location[0] - location[2]/2
        location[1] = location[1] - location[3]/2
        loss= sum([(location[i] - gt_location[i])**2 for i in range(4)]) * 100 / 4
        return loss

    def cal_yolo_IOU(self, location, gt_location):
        # Translate yolo's box mid-point (x0, y0) to top-left point (x1, y1), in order to compare with gt
        location[0] = location[0] - location[2]/2
        location[1] = location[1] - location[3]/2
        loss = self.iou(location, gt_location)
        return loss


    def prepare_training_data(self, img_fold, gt_file, out_fold, without_gt = False):  #[or]prepare_training_data(self, list_file, gt_file, out_fold):
        ''' Pass the data through YOLO, and get the fc_17 layer as features, and get the fc_19 layer as locations
         Save the features and locations into file for training LSTM'''
        # Reshape the input image
        paths= self.load_folder(img_fold)
        if(without_gt == False):
            gt_locations= self.load_dataset_gt(gt_file)

        avg_loss = 0
        total= 0
        total_time= 0

        for id, path in enumerate(paths):
            filename= os.path.basename(path)
            print("processing: ", id, ": ", filename)
            img = self.file_to_img(path)

            # Pass through YOLO layers
            self.h_img,self.w_img,_ = img.shape
            img_resized = cv2.resize(img, (448, 448))
            img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
            img_resized_np = np.asarray( img_RGB )
            inputs = np.zeros((1,448,448,3),dtype='float32')
            inputs[0] = (img_resized_np/255.0)*2.0-1.0
            in_dict = {self.x : inputs}

            start_time = time.time()
            feature= self.sess.run(self.fc_30,feed_dict=in_dict)
            cycle_time = time.time() - start_time
            print('cycle time= ', cycle_time)
            total_time += cycle_time
            output = self.sess.run(self.fc_32,feed_dict=in_dict)  # make sure it does not run conv layers twice

            locations = self.interpret_output(output[0])
            if(without_gt == False):
                gt_location = self.find_gt_location(gt_locations, id)
                location = self.find_best_location(locations, gt_location) # find the ROI that has the maximum IOU with the ground truth
    
                self.debug_location(img, location)
                self.debug_gt_location(img, gt_location)

                # change location into [0, 1]
                loss= self.cal_yolo_IOU(location[1:5], gt_location)
                location = self.location_from_0_to_1(self.w_img, self.h_img, location)
                avg_loss += loss
                print("loss: ", loss)
                total += 1
                yolo_output=  np.concatenate(
                                      ( np.reshape(feature, [-1, self.num_feat]),
                                        np.reshape(location, [-1, self.num_predict]) ),
                                      axis = 1)
            else:
                total += 1
                yolo_output=  np.reshape(feature, [-1, self.num_feat])
            self.save_yolo_output(out_fold, yolo_output, filename)

        avg_loss = avg_loss/total
        print("YOLO avg_loss: ", avg_loss)

        print "Time Spent on Tracking: " + str(total_time)
        print "fps: " + str(id/total_time)

        return


    def loc_to_coordinates(self, loc):
        loc = [i * 32 for i in loc]
        x1= int(loc[0]- loc[2]/2)
        y1= int(loc[1]- loc[3]/2)
        x2= int(loc[0]+ loc[2]/2)
        y2= int(loc[1]+ loc[3]/2)
        return [x1, y1, x2, y2]


    def coordinates_to_heatmap_vec(self, coord):
        heatmap_vec = np.zeros(1024)
        print(coord)
        [classnum, x1, y1, x2, y2, prob] = coord
        [x1, y1, x2, y2]= self.loc_to_coordinates([x1, y1, x2, y2])
        for y in range(y1, y2):
            for x in range(x1, x2):
                index = y*32 + x
                heatmap_vec[index] = 1.0   
        return heatmap_vec


    def prepare_training_data_heatmap(self, img_fold, gt_file, out_fold):  #[or]prepare_training_data(self, list_file, gt_file, out_fold):
        ''' Pass the data through YOLO, and get the fc_17 layer as features, and get the fc_19 layer as locations
         Save the features and locations into file for training LSTM'''
        # Reshape the input image
        paths= self.load_folder(img_fold)
        gt_locations= self.load_dataset_gt(gt_file)

        avg_loss = 0
        total= 0

        for id, path in enumerate(paths):
            filename= os.path.basename(path)
            print("processing: ", id, ": ", filename)
            img = self.file_to_img(path)

            # Pass through YOLO layers
            self.h_img,self.w_img,_ = img.shape
            img_resized = cv2.resize(img, (448, 448))
            img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
            img_resized_np = np.asarray( img_RGB )
            inputs = np.zeros((1,448,448,3),dtype='float32')
            inputs[0] = (img_resized_np/255.0)*2.0-1.0

            in_dict = {self.x : inputs}
            feature= self.sess.run(self.fc_30,feed_dict=in_dict)
            output = self.sess.run(self.fc_32,feed_dict=in_dict)  # make sure it does not run conv layers twice
            locations = self.interpret_output(output[0])
            gt_location = self.find_gt_location(gt_locations, id)
            location = self.find_best_location(locations, gt_location) # find the ROI that has the maximum IOU with the ground truth

            self.debug_location(img, location)
            self.debug_gt_location(img, gt_location)

            # change location into [0, 1]
            loss= self.cal_yolo_IOU(location[1:5], gt_location)

            location = self.location_from_0_to_1(self.w_img, self.h_img, location)
            heatmap_vec= self.coordinates_to_heatmap_vec(location)

            avg_loss += loss
            total += 1
            print("loss: ", loss)

            yolo_output=  np.concatenate(
                                  ( np.reshape(feature, [-1, self.num_feat]),
                                    np.reshape(heatmap_vec, [-1, self.num_heatmap]) ),
                                  axis = 1)

            self.save_yolo_output(out_fold, yolo_output, filename)

        avg_loss = avg_loss/total
        print("YOLO avg_loss: ", avg_loss)

        return


    def prepare_training_data_multiTarget(self, img_fold, out_fold):
        ''' Pass the data through YOLO, and get the fc_17 layer as features, and get the fc_19 layer as locations
         Save the features and locations into file for training LSTM'''
        # Reshape the input image
        print(img_fold)
        paths= self.load_folder(img_fold)
        avg_loss = 0
        total= 0

        for id, path in enumerate(paths):
            filename= os.path.basename(path)
            print("processing: ", id, ": ", filename)
            img = self.file_to_img(path)

            # Pass through YOLO layers
            self.h_img,self.w_img,_ = img.shape

            img_resized = cv2.resize(img, (448, 448))
            img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
            img_resized_np = np.asarray( img_RGB )
            inputs = np.zeros((1,448,448,3),dtype='float32')
            inputs[0] = (img_resized_np/255.0)*2.0-1.0

            in_dict = {self.x : inputs}
            feature= self.sess.run(self.fc_30,feed_dict=in_dict)
            output = self.sess.run(self.fc_32,feed_dict=in_dict)  # make sure it does not run conv layers twice
            locations = self.interpret_output(output[0])
            self.debug_locations(img, locations)

            # change location into [0, 1]
            for i in range(0, len(locations)):
                class_index = self.classes.index(locations[i][0])
                locations[i][0] = class_index
                locations[i] = self.location_from_0_to_1(self.w_img, self.h_img, locations[i])
            if len(locations)== 1:
                print('len(locations)= 1\n')
                yolo_output = [[np.reshape(feature, [-1, self.num_feat])], [np.reshape(locations, [-1, self.num_predict]), [0,0,0,0,0,0]]]
            else:
                yolo_output = [[np.reshape(feature, [-1, self.num_feat])], [np.reshape(locations, [-1, self.num_predict])]]
            self.save_yolo_output(out_fold, yolo_output, filename)

        return

'''----------------------------------------main-----------------------------------------------------'''
def main(argv):
    ''' PARAMETERS '''
    num_steps= 6
    test = 29
    newFps = 5
    sequence_name = 'Car4_2'
    dataFolder = os.path.join('benchmark/DATA/',sequence_name)
    
    '''
    yolo_output = np.load('benchmark/DATA/Car4/rolo_out_test/655.npy')
    rolo_output = np.load('benchmark/DATA/Car4_2/rolo_out_test/655.npy')
    print(yolo_output,'yolo_output',rolo_output,'rolo_output')
    exit(1)
    '''
    img_fold_path = os.path.join(dataFolder, 'img/')
    gt_file_path= os.path.join(dataFolder, 'groundtruth_rect.txt')
    yolo_out_path= os.path.join(dataFolder, 'yolo_out/')
    rolo_out_path= os.path.join(dataFolder, 'rolo_out_test/')

    #convert video to images
    if(not os.path.exists(os.path.join(dataFolder, 'img', '0001.jpg'))):
        vidcap = cv2.VideoCapture(dataFolder + '.mp4')
        if(not os.path.exists(dataFolder)):
            os.mkdir(dataFolder)
        if(not os.path.exists(os.path.join(dataFolder, 'img'))):
            os.mkdir(os.path.join(dataFolder, 'img'))
        if(not os.path.exists(os.path.join(dataFolder, 'rolo_out_test'))):
            os.mkdir(os.path.join(dataFolder, 'rolo_out_test'))
        success,image = vidcap.read()
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver)  < 3 :
            fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
            print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
        else :
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)
        saveEachFrame = int(round(fps/newFps))
        print('I am saving just 1 frame out of ' + str(saveEachFrame))
        count = 180
        success = True
        frameNum = 1
        while success:
            success,image = vidcap.read()
            if(success == True and frameNum%saveEachFrame == 0):
                cv2.imwrite(os.path.join(dataFolder, 'img', "%04d.jpg" % count), image)     # save frame as JPEG file
                count += 1
            frameNum += 1
    #if still images do not exists, then throw the error
    if(not os.path.exists(os.path.join(dataFolder, 'img', '0001.jpg'))): 
        print('An error occured. I cannot find an images with in the folder ('+os.path.join(dataFolder, 'img', '00001.jpg')+')')
        exit(1)
    img = cv2.imread(os.path.join(dataFolder, 'img', '0001.jpg'))
    ht, wid, _ = img.shape
    
    #get YOLO features
    yolo = YOLO_TF(['YOLO/YOLO_network.py'])
    yolo.w_img = wid
    yolo.h_img = ht
    [yolo.w_img, yolo.h_img, sequence_name, dummy_1, dummy_2]= util.choose_video_sequence(test)
    out_fold = os.path.join(dataFolder, 'yolo_out/')
    yolo.createFolder(out_fold)
    yolo.prepare_training_data(os.path.join(dataFolder, 'img/'),'',out_fold, without_gt = True)
    #yolo.prepare_training_data_multiTarget(img_fold,out_fold)
    '''
    #get ROLO output
    ROLO_net = ROLO_TF({'test_folder':sequence_name})
    paths_imgs = utils.load_folder( img_fold_path)
    paths_rolo= utils.load_folder( rolo_out_path)

    # Define the codec and create VideoWriter object
    fourcc= cv2.VideoWriter_fourcc(*'DIVX')
    video_name = sequence_name + '_test.avi'
    video_path = os.path.join('output/videos/', video_name)
    video = cv2.VideoWriter(video_path, fourcc, 20, (wid, ht))


    total= 0
    rolo_avgloss= 0
    yolo_avgloss= 0
    for i in range(len(paths_rolo)- num_steps):
        id= i + 1
        test_id= id + num_steps - 2  #* num_steps + 1

        path = paths_imgs[test_id]
        img = utils.file_to_img(path)

        if(img is None): break

        yolo_location= utils.find_yolo_location(yolo_out_path, test_id)
        yolo_location= utils.locations_normal( wid, ht, yolo_location)
        print(yolo_location)

        rolo_location= utils.find_rolo_location( rolo_out_path, test_id)
        rolo_location = utils.locations_normal( wid, ht, rolo_location)
        print(rolo_location)

        #gt_location = utils.find_gt_location( lines, test_id - 1)
        #gt_location= locations_from_0_to_1(None, 480, 640, gt_location)
        #gt_location = locations_normal(None, 480, 640, gt_location)
        print('gt: ' + str(test_id))
        #print(gt_location)

        frame = utils.debug_2_locations( img, yolo_location, rolo_location)
        video.write(frame)

        utils.createFolder(os.path.join('output/frames/',sequence_name))
        frame_name= os.path.join('output/frames/',sequence_name,str(test_id)+'.jpg')
        print(frame_name)
        cv2.imwrite(frame_name, frame)
        #cv2.imshow('frame',frame)
        #cv2.waitKey(100)
        total += 1
    video.release()
    cv2.destroyAllWindows()
    '''

if __name__=='__main__':
    main(sys.argv)
