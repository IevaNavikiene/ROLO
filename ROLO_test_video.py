'''
Script File: ROLO_test_video.py

Description:

    This script helps to test ROLO on your video (.mp4) files without any additional knowledge
'''

import cv2
import os
import numpy as np
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import time
import random
#just for visualization
sys.path.insert(0, 'utils')
import ROLO_utils as utils

class ROLO_YOLO():
    imshow = True
    disp_console = True
    weights_file = 'weights/YOLO_small.ckpt'
    alpha = 0.1
    w_img = 0
    h_img = 0#[argvs['height'], argvs['width']]
    num_feat = 4096
    num_predict = 6 # final output of LSTM 6 loc parameters
    
    # ROLO Network Parameters
    num_steps = 3  # number of frames as an input sequence
    num_input = num_feat + num_predict # data input: 4096+6= 5002
    rolo_weights_file= '/home/ieva/projects/ROLO/experiments/testing/model_step3_exp3.ckpt' 
    # ROLO Parameters
    batch_size = 1
    
    def __init__(self, argvs=[], folder_name='', out_fold=''):
        self.argv_parser(argvs)
        self.folder_name = folder_name
        self.out_fold = out_fold
        self.rolo_output_path = os.path.join('benchmark/DATA', folder_name, 'rolo_out_test/')
        self.initializeROLONetwork()
        self.initializeNetwork()

    def initializeNetwork(self):
        YOLO_graph = tf.Graph()
        with YOLO_graph.as_default():
            def conv_layer(idx,inputs,filters,size,stride):
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

            def pooling_layer(idx,inputs,size,stride):
                if self.disp_console : print '    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx,size,size,stride)
                return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')

            def fc_layer(idx,inputs,hiddens,flat = False,linear = False):
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
            if self.disp_console : print "Building YOLO_small graph..."
            self.x = tf.placeholder('float32',[None,448,448,3])
            conv_1 = conv_layer(1,self.x,64,7,2)
            pool_2 = pooling_layer(2,conv_1,2,2)
            conv_3 = conv_layer(3,pool_2,192,3,1)
            pool_4 = pooling_layer(4,conv_3,2,2)
            conv_5 = conv_layer(5,pool_4,128,1,1)
            conv_6 = conv_layer(6,conv_5,256,3,1)
            conv_7 = conv_layer(7,conv_6,256,1,1)
            conv_8 = conv_layer(8,conv_7,512,3,1)
            pool_9 = pooling_layer(9,conv_8,2,2)
            conv_10 = conv_layer(10,pool_9,256,1,1)
            conv_11 = conv_layer(11,conv_10,512,3,1)
            conv_12 = conv_layer(12,conv_11,256,1,1)
            conv_13 = conv_layer(13,conv_12,512,3,1)
            conv_14 = conv_layer(14,conv_13,256,1,1)
            conv_15 = conv_layer(15,conv_14,512,3,1)
            conv_16 = conv_layer(16,conv_15,256,1,1)
            conv_17 = conv_layer(17,conv_16,512,3,1)
            conv_18 = conv_layer(18,conv_17,512,1,1)
            conv_19 = conv_layer(19,conv_18,1024,3,1)
            pool_20 = pooling_layer(20,conv_19,2,2)
            conv_21 = conv_layer(21,pool_20,512,1,1)
            conv_22 = conv_layer(22,conv_21,1024,3,1)
            conv_23 = conv_layer(23,conv_22,512,1,1)
            conv_24 = conv_layer(24,conv_23,1024,3,1)
            conv_25 = conv_layer(25,conv_24,1024,3,1)
            conv_26 = conv_layer(26,conv_25,1024,3,2)
            conv_27 = conv_layer(27,conv_26,1024,3,1)
            conv_28 = conv_layer(28,conv_27,1024,3,1)
            fc_29 = fc_layer(29,conv_28,512,flat=True,linear=False)
            self.fc_30 = fc_layer(30,fc_29,4096,flat=False,linear=False)
            #skip dropout_31
            fc_32 = fc_layer(32,self.fc_30,1470,flat=False,linear=True)
        self.YOLOSession = tf.Session(graph=YOLO_graph)
        with self.YOLOSession as sessYOLO:
            sessYOLO.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            print(self.weights_file,'weights_file')
            saver.restore(sessYOLO,self.weights_file)
            if self.disp_console : print "Loading complete!" + '\n'
            self.prepare_training_data(os.path.join(self.folder_name, 'img/'), self.out_fold)
        
    def argv_parser(self, argvs):
        for i in range(1,len(argvs),2):
            if argvs[i] == 'width' : w_img = argvs[i+1] ;
            if argvs[i] == 'height' : h_img = argvs[i+1] ;
            if argvs[i] == 'num_steps' : num_steps = argvs[i+1] ;
            if argvs[i] == '-imshow' :
                if argvs[i+1] == '1' :imshow = True
                else : imshow = False
            if argvs[i] == '-disp_console' :
                if argvs[i+1] == '1' :disp_console = True
                else : disp_console = False        

    def prepare_training_data(self, img_fold, out_fold):  #[or]prepare_training_data( list_file, gt_file, out_fold):
        
        def file_to_img(filepath):
            img = cv2.imread(filepath)
            return img
        
        def load_folder(path):
            paths = [os.path.join(path,fn) for fn in next(os.walk(path))[2]]
            #return paths
            return sorted(paths)  
        ''' Pass the data through YOLO, and get the fc_17 layer as features, and get the fc_19 layer as locations
         Save the features and locations into file for training LSTM'''
        # Reshape the input image
        paths= load_folder(img_fold)
        avg_loss = 0
        total= 0
        total_time= 0
        if(not os.path.exists(out_fold)):
            os.mkdir(out_fold)
        YOLO_features = np.empty((1, self.num_steps, 4102))
        roloId= 0
        for id, path in enumerate(paths):
            filename= os.path.basename(path)
            print("processing: ", id, ": ", filename)
            img = file_to_img(path)
            # Pass through YOLO layers
            h_img,w_img,_ = img.shape
            img_resized = cv2.resize(img, (448, 448))
            img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
            img_resized_np = np.asarray( img_RGB )
            inputs = np.zeros((1,448,448,3),dtype='float32')
            inputs[0] = (img_resized_np/255.0)*2.0-1.0
            in_dict = {self.x : inputs}

            start_time = time.time()
            #GET features
            feature= self.YOLOSession.run(self.fc_30,feed_dict=in_dict)
            cycle_time = time.time() - start_time
            print('cycle time= ', cycle_time)
            #output = sess.run(fc_32,feed_dict=in_dict)  # make sure it does not run conv layers twice
            total += 1
            #yolo_output=  np.reshape(feature, [-1, self.num_feat])
            print(feature.shape,'tttttttttt')
            YOLO_features[:,roloId,0:self.num_feat] = feature
            roloId += 1
            if(total%self.num_steps == 0):
                print('Trying 3 frames with ROLO')
                self.runROLOonYOLO(YOLO_features, id)
                roloId = 0
            #save_yolo_output(out_fold, yolo_output, filename)
        print "Time Spent on Tracking: " + str(total_time)
        print "fps: " + str(id/total_time)
        return

    def save_yolo_output( out_fold, yolo_output, filename):
        name_no_ext= os.path.splitext(filename)[0]
        output_name= name_no_ext
        path = os.path.join(out_fold, output_name)
        np.save(path, yolo_output)                                                
    
    def initializeROLONetwork(self):
        self.ROLO_graph = tf.Graph()
        with self.ROLO_graph.as_default():
            # tf Graph input
            self.x_rolo = tf.placeholder("float32", [None, self.num_steps, self.num_input])
            istate = tf.placeholder("float32", [None, 2*self.num_input]) #state & cell => 2x num_input

            # Define weights
            weights = {
                'out': tf.Variable(tf.random_normal([self.num_input, self.num_predict]))
            }
            biases = {
                'out': tf.Variable(tf.random_normal([self.num_predict]))
            }

            def LSTM_single(name,  _X, _weights, _biases):
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
                    cell = rnn.BasicLSTMCell(self.num_input, self.num_input)#tf.nn.rnn_cell.LSTMCell(num_input, num_input)            
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
                return outputs, state

            #rolo_utils= utils.ROLO_utils()
            #rolo_utils.loadCfg()
            #Initialize all ROLO session
            if self.disp_console : print "Building ROLO graph..."

            # Build rolo layers
            pred, istate = LSTM_single('lstm_test', self.x_rolo, weights, biases)
            ious= tf.Variable(tf.zeros([self.batch_size]), name="ious")
            self.prediction_location = pred[0][:, 4097:4101]
        #self.ROLOsession = tf.Session(graph=ROLO_graph)
        with tf.Session(graph=self.ROLO_graph) as sessROLO:
            sessROLO.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            saver.restore(sessROLO, self.rolo_weights_file)
            print "Loading complete!" + '\n'
            
    def runROLOonYOLO(self, YOLO_features, id):
        with tf.Session(graph=self.ROLO_graph) as sessROLO:
            batch_xs = np.reshape(YOLO_features, [self.batch_size, self.num_steps, self.num_input])
            pred_location= sessROLO.run(self.prediction_location,feed_dict={self.x_rolo: batch_xs})
            # Save pred_location to file
            utils.save_rolo_output_test(self.rolo_output_path, pred_location, id, self.num_steps, self.batch_size)
    

'''----------------------------------------main-----------------------------------------------------'''
def main(argv):
    ''' PARAMETERS '''
    num_steps= 3
    newFps = 5
    sequence_name = 'Car4_2'
    dataFolder = os.path.join('benchmark/DATA/',sequence_name)
    img_fold_path = os.path.join(dataFolder, 'img/')
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
    if(not os.path.exists(rolo_out_path)):
        os.mkdir(rolo_out_path)

    #Get image width and height    
    img = cv2.imread(os.path.join(dataFolder, 'img', '0001.jpg'))
    ht, wid, _ = img.shape
    
    #get YOLO features
    yolo = ROLO_YOLO(['height',wid, 'width',ht,'num_steps',num_steps], dataFolder,yolo_out_path)

    paths_imgs = utils.load_folder(img_fold_path)
    paths_rolo= utils.load_folder(rolo_out_path)

    # Define the codec and create VideoWriter object
    fourcc= cv2.VideoWriter_fourcc(*'DIVX')
    video_name = sequence_name + '_test.avi'
    video_path = os.path.join('output/videos/', video_name)
    video = cv2.VideoWriter(video_path, fourcc, 20, (wid, ht))
    total= 0

    for i in range(len(paths_rolo)- num_steps):
        id= i + 1
        test_id= id + num_steps - 2  #* num_steps + 1

        path = paths_imgs[test_id]
        img = utils.file_to_img(path)

        if(img is None): break

        yolo_location= utils.find_yolo_location(yolo_out_path, test_id)
        yolo_location= utils.locations_normal( wid, ht, yolo_location)
        rolo_location= utils.find_rolo_location( rolo_out_path, test_id)
        rolo_location = utils.locations_normal( wid, ht, rolo_location)
        print('gt: ' + str(test_id))
        frame = utils.debug_2_locations( img, yolo_location, rolo_location)
        video.write(frame)
        utils.createFolder(os.path.join('output/frames/',sequence_name))
        frame_name= os.path.join('output/frames/',sequence_name,str(test_id)+'.jpg')
        cv2.imwrite(frame_name, frame)
        #cv2.imshow('frame',frame)
        #cv2.waitKey(100)
        total += 1
    video.release()
    cv2.destroyAllWindows()
    

if __name__=='__main__':
    main(sys.argv)
