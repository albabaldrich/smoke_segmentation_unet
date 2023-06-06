import numpy as np
import cv2    
import matplotlib.pyplot as plt
import torch
import glob
import os


def ExtractFrames(cfg):
    #Load video path
    video_path = os.path.join(cfg['video_dataset'], cfg['video_name']  + cfg['form_video'])

    capture = cv2.VideoCapture(video_path) #read the video from the path
    n = 0
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    
    while (True):
        ret, frame = capture.read() #read the video frame by frame
        #print(frame.shape)
        if ret == False:
            break
        else:
            if n < 10:
                frame_name = os.path.join(cfg['output'], cfg['video_name'], 'frames', 'frame_0000' + str(n) + cfg['form_img'])  # frame path + frame name
            elif n < 100:
                frame_name = os.path.join(cfg['output'], cfg['video_name'], 'frames', 'frame_000' + str(n) + cfg['form_img'])  # frame path + frame name
            elif n < 1000:
                    frame_name = os.path.join(cfg['output'], cfg['video_name'], 'frames', 'frame_00' + str(n) + cfg['form_img'])  # frame path + frame name
            elif n < 10000:
                frame_name = os.path.join(cfg['output'], cfg['video_name'], 'frames', 'frame_0' + str(n) + cfg['form_img'])  # frame path + frame name
            else:
                frame_name = os.path.join(cfg['output'], cfg['video_name'], 'frames', 'frame_' + str(n) + cfg['form_img'])  # frame path + frame name
            
           # crop_frame = frame[start_x:end_x, start_y:end_y]
            #cv2.imwrite(frame_name, crop_frame)  # save the frame in this path
            cv2.imwrite(frame_name, frame)  # save the frame in this path
            print('Saving frame ' + str(n))  #print how many frames are created
            n += 1

    capture.release()


def VideoPred (cfg):
    #Load model and frames dataset
    frame_path = os.path.join(cfg['output'], cfg['video_name'], 'frames')
    model_load = os.path.join(cfg['model_path'], model_name)

