import argparse
import cv2 as cv
import numpy as np
import time

parser = argparse.ArgumentParser(description="Live inference with YOLOv8.")
parser.add_argument('--webcam_index', type=int, default=0, help='Index of the webcam to use.')
args = parser.parse_args()

import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors 

model_seg = YOLO('yolov8n-seg.pt', task='segment')
model_pose = YOLO('yolov8n-pose.pt', task='pose') 
colors_palette = Colors() 

cap = cv.VideoCapture(args.webcam_index)

def fps(start, end):
    return int(1//(end-start))

try:
    screen_width = 1920  
    screen_height = 1080  
    
    try:
        screen = cv.getWindowImageRect('Multi-view Display')
        if screen:
            screen_width, screen_height = screen[2], screen[3]
    except:
        pass
    
    grid_width = screen_width
    grid_height = int(grid_width * 9/16 * 2)  
    
    if grid_height > screen_height:
        grid_height = screen_height
        grid_width = int(grid_height * 16/9 / 2 * 2)  
    
    panel_width = grid_width // 2
    panel_height = grid_height // 2

    device = 0 if torch.cuda.is_available() else 'cpu'

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            print('No camera detected, aborting')
            break
        image = cv.flip(image, 1)
        start = time.perf_counter()
        results = model_seg.predict(image, classes = [0, 16, 24, 25, 26, 39, 41, 47, 56, 62, 67, 73, 74], device=device) 
        results_pose = model_pose.predict(image) 
        end = time.perf_counter()
        
        # Screen 1: Raw image
        plotted_image_no_boxes =  image.copy()
        cv.putText(plotted_image_no_boxes, f'FPS: {fps(start, end)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Screen 2: Just the masks
        h_display, w_display, _ = plotted_image_no_boxes.shape
        combined_mask_display_bgr = np.zeros((h_display, w_display, 3), dtype=np.uint8)
        cv.putText(combined_mask_display_bgr, f'Segmentacao', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if results[0].masks is not None and results[0].boxes is not None:
            masks_data_cpu = results[0].masks.data.cpu().numpy()  
            
            for i in range(len(results[0].boxes)): 
                if i < len(masks_data_cpu): 
                    cls_index = int(results[0].boxes[i].cls[0])
                    color = colors_palette(cls_index, True)  
                    
                    object_mask_model_res = masks_data_cpu[i]
                    object_mask_display_res = cv.resize(object_mask_model_res, (w_display, h_display), interpolation=cv.INTER_LINEAR)
                                        
                    binary_mask_for_coloring = (object_mask_display_res > 0.5) 
                    combined_mask_display_bgr[binary_mask_for_coloring] = color
        else:
            pass
          
        # Screen 3: Object detection with bounding boxes
        plotted_image_with_boxes = results[0].plot(boxes=True)
        cv.putText(plotted_image_with_boxes, "Deteccao", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Screen 4: Pose Estimation on black background, no boxes
        h_target, w_target = plotted_image_no_boxes.shape[:2]
        black_background_pose = np.zeros((h_target, w_target, 3), dtype=np.uint8)
        
        if results_pose and results_pose[0].keypoints is not None:     
            plotted_pose_image = results_pose[0].plot(img=black_background_pose, boxes=False, kpt_radius=5, kpt_line=True)
        else:
            plotted_pose_image = black_background_pose.copy()
        cv.putText(plotted_pose_image, "Faca uma pose", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        src_h, src_w = image.shape[:2]
        src_ratio = src_w / src_h
        target_ratio = 16/9
    
        if src_ratio > target_ratio:  
            resize_w = panel_width
            resize_h = int(panel_width / src_ratio)
            top_padding = (panel_height - resize_h) // 2
            bottom_padding = panel_height - resize_h - top_padding
            left_padding = 0
            right_padding = 0
        else:  
            resize_h = panel_height
            resize_w = int(panel_height * src_ratio)            
            left_padding = (panel_width - resize_w) // 2
            right_padding = panel_width - resize_w - left_padding
            top_padding = 0
            bottom_padding = 0
        
        plotted_image_no_boxes = cv.resize(plotted_image_no_boxes, (resize_w, resize_h))
        combined_mask_display_bgr = cv.resize(combined_mask_display_bgr, (resize_w, resize_h))
        plotted_image_with_boxes = cv.resize(plotted_image_with_boxes, (resize_w, resize_h))
        plotted_pose_image = cv.resize(plotted_pose_image, (resize_w, resize_h))
        
        plotted_image_no_boxes = cv.copyMakeBorder(plotted_image_no_boxes, top_padding, bottom_padding,
                                                left_padding, right_padding, cv.BORDER_CONSTANT, value=[0, 0, 0])
        combined_mask_display_bgr = cv.copyMakeBorder(combined_mask_display_bgr, top_padding, bottom_padding,
                                                    left_padding, right_padding, cv.BORDER_CONSTANT, value=[0, 0, 0])
        plotted_image_with_boxes = cv.copyMakeBorder(plotted_image_with_boxes, top_padding, bottom_padding,
                                                    left_padding, right_padding, cv.BORDER_CONSTANT, value=[0, 0, 0])
        plotted_pose_image = cv.copyMakeBorder(plotted_pose_image, top_padding, bottom_padding,
                                            left_padding, right_padding, cv.BORDER_CONSTANT, value=[0, 0, 0])
        
        top_row = cv.hconcat([plotted_image_no_boxes, combined_mask_display_bgr])
        bottom_row = cv.hconcat([plotted_image_with_boxes, plotted_pose_image])
        grid_display = cv.vconcat([top_row, bottom_row])
        
        cv.namedWindow('Multi-view Display', cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty('Multi-view Display', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow('Multi-view Display', grid_display)

        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            print('Exit sequence initiated')
            break

finally:
    cap.release()
    cv.destroyAllWindows()
