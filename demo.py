import argparse
import pathlib
import numpy as np
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from PIL import Image, ImageOps

from face_detection import RetinaFace

from l2cs import select_device, draw_gaze, getArch, Pipeline, render, find_border_points

CWD = pathlib.Path.cwd()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--device',dest='device', help='Device to run model: cpu or gpu:0',
        default="cpu", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

def calibration(width, height):
    calibration_step_time = 8
    instruction_time = 12
    calibration_point_cnt = 4
    calibration_point_size = 25
    safe_area = [int(calibration_point_size), int(width-calibration_point_size), int(height-calibration_point_size)]
    calibration_points_places = [[0,0], [1,0], [1,2], [0,2]]
    points = [[], [], [], []]
    help_message_1 = "Please be ready for the calibration process. Calibration is done in "+str(calibration_point_cnt)+" steps and"
    help_message_2 = "each step takes "+str(calibration_step_time)+" seconds. Please look at the red point in each step."
    messages = [help_message_1, help_message_2]
    for i in range(calibration_point_cnt):
        messages.append("Please look at the point No. " + str(i+1) + " for "+ str(calibration_step_time)+" seconds.")

    start_time = time.time()
    while time.time() - start_time < (instruction_time+calibration_step_time*calibration_point_cnt):
        success, frame = cap.read()
        if not success:
            print("Failed to obtain frame")
            time.sleep(0.1)
        frame = cv2.flip(frame, 1)
        results = gaze_pipeline.step(frame)
        _, dx, dy = render(frame, results, width, height, [], True)

        if time.time() - start_time < instruction_time:
            cv2.putText(frame, messages[0], (100, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, messages[1], (100, 50),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1, cv2.LINE_AA)
            for i in range(calibration_point_cnt):
                cv2.circle(frame, (safe_area[calibration_points_places[i][0]],safe_area[calibration_points_places[i][1]]), calibration_point_size, (0, 255, 0), -1)
                cv2.putText(frame, str(i+1), (safe_area[calibration_points_places[i][0]],safe_area[calibration_points_places[i][1]]),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1, cv2.LINE_AA)
        else:
            i = min(int(int(time.time() - start_time - instruction_time) / calibration_step_time), calibration_point_cnt-1)
            cv2.putText(frame, messages[2+i], (300, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, (safe_area[calibration_points_places[i][0]],safe_area[calibration_points_places[i][1]]), calibration_point_size, (0, 0, 255), -1)
            # cv2.putText(frame, str(i+1), (safe_area[calibration_points_places[i][0]],safe_area[calibration_points_places[i][1]]),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
            points[i].append([dx, dy])

        cv2.imshow("Demo",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        success,frame = cap.read()

    print("\npoints=", points)

    method="avg"
    # method="max"
    return find_border_points(points, method)

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    cam = args.cam_id
    # snapshot_path = args.snapshot

    gaze_pipeline = Pipeline(
        weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device = select_device(args.device, batch_size=1)
    )
     
    cap = cv2.VideoCapture(cam)
    
    width, height  = cap.get(3), cap.get(4)
    print("width:", width, " height:", height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width, height  = cap.get(3), cap.get(4)
    print("width:", width, " height:", height)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        border_points = calibration(width, height)
        print("\nborder_points=",border_points)
        while True:

            # Get frame
            success, frame = cap.read()    
            start_fps = time.time()  

            if not success:
                print("Failed to obtain frame")
                time.sleep(0.1)

            frame = cv2.flip(frame, 1)

            # Process frame
            results = gaze_pipeline.step(frame)

            # Visualize output
            frame, _, _ = render(frame, results, width, height, border_points, False)
           
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            success,frame = cap.read()  
    
