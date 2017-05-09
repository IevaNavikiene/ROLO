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
Script File: ROLO_demo_test.py

Description:

    ROLO is short for Recurrent YOLO, aimed at simultaneous object detection and tracking
    Paper: http://arxiv.org/abs/1607.05781
    Author: Guanghan Ning
    Webpage: http://guanghan.info/
'''

import cv2
import os
import numpy as np
import sys
sys.path.insert(0, 'utils')
import ROLO_utils as utils
'''----------------------------------------main-----------------------------------------------------'''
def main(argv):
    ''' PARAMETERS '''
    num_steps= 6
    test = 29
    newFps = 5
    wid, ht = [640, 480]
    sequence_name = 'Strange'

    img_fold_path = os.path.join('benchmark/DATA', sequence_name, 'img/')
    gt_file_path= os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
    yolo_out_path= os.path.join('benchmark/DATA', sequence_name, 'yolo_out/')
    rolo_out_path= os.path.join('benchmark/DATA', sequence_name, 'rolo_out_test/')

    #convert video to images
    if(not os.path.exists(os.path.join('benchmark/DATA', sequence_name, 'img', '00001.jpg'))):
        vidcap = cv2.VideoCapture(os.path.join('benchmark/DATA', sequence_name + '.mp4'))
        if(not os.path.exists(os.path.join('benchmark/DATA', sequence_name))):
            os.mkdir(os.path.join('benchmark/DATA', sequence_name))
        if(not os.path.exists(os.path.join('benchmark/DATA', sequence_name, 'img'))):
            os.mkdir(os.path.join('benchmark/DATA', sequence_name, 'img'))
        if(not os.path.exists(os.path.join('benchmark/DATA', sequence_name, 'rolo_out_test'))):
            os.mkdir(os.path.join('benchmark/DATA', sequence_name, 'rolo_out_test'))
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
                cv2.imwrite(os.path.join('benchmark/DATA', sequence_name, 'img', "%05d.jpg" % count), image)     # save frame as JPEG file
                count += 1
            frameNum += 1
            
    paths_imgs = utils.load_folder( img_fold_path)
    paths_rolo= utils.load_folder( rolo_out_path)
    lines = utils.load_dataset_gt( gt_file_path)

    # Define the codec and create VideoWriter object
    fourcc= cv2.VideoWriter_fourcc(*'DIVX')
    video_name = sequence_name + '_test.avi'
    video_path = os.path.join('output/videos/', video_name)
    video = cv2.VideoWriter(video_path, fourcc, 20, (wid, ht))


    total= 0
    rolo_avgloss= 0
    yolo_avgloss= 0
    print(len(paths_rolo),num_steps, 'tttttttttttt',rolo_out_path,'rolo_out_path')
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

        gt_location = utils.find_gt_location( lines, test_id - 1)
        #gt_location= locations_from_0_to_1(None, 480, 640, gt_location)
        #gt_location = locations_normal(None, 480, 640, gt_location)
        print('gt: ' + str(test_id))
        print(gt_location)

        frame = utils.debug_3_locations( img, gt_location, yolo_location, rolo_location)
        video.write(frame)

        utils.createFolder(os.path.join('output/frames/',sequence_name))
        frame_name= os.path.join('output/frames/',sequence_name,str(test_id)+'.jpg')
        print(frame_name)
        cv2.imwrite(frame_name, frame)
        #cv2.imshow('frame',frame)
        #cv2.waitKey(100)

        rolo_loss = utils.cal_rolo_IOU(rolo_location, gt_location)
        rolo_avgloss += rolo_loss
        yolo_loss=  utils.cal_yolo_IOU(yolo_location, gt_location)
        yolo_avgloss += yolo_loss
        total += 1
    if(total != 0):
        rolo_avgloss /= total
        yolo_avgloss /= total
    print("yolo_avg_iou = ", yolo_avgloss)
    print("rolo_avg_iou = ", rolo_avgloss)
    video.release()
    cv2.destroyAllWindows()
    



if __name__=='__main__':
    main(sys.argv)
