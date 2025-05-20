# from PIL import Image
# import cv2
# import torch
# import math 
# import function.utils_rotate as utils_rotate
# from IPython.display import display
# import os
# import time
# import argparse
# import function.helper as helper

# # Load YOLO models
# yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
# yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
# yolo_license_plate.conf = 0.60

# prev_frame_time = 0
# new_frame_time = 0

# # Open video file
# vid = cv2.VideoCapture('/Users/kaiser_1/Documents/Zalo Received Files/QNO-6082R_20250325175117.avi')

# # Get video properties
# frame_width = int(vid.get(3))
# frame_height = int(vid.get(4))
# fps = int(vid.get(cv2.CAP_PROP_FPS))

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
# video_writer = cv2.VideoWriter('/Users/kaiser_1/Documents/GitHub/License-Plate-Recognition/output_mp4/output3.mp4', fourcc, fps, (frame_width, frame_height))

# while True:
#     ret, frame = vid.read()
#     if not ret:
#         break  # Exit if video ends
    
#     plates = yolo_LP_detect(frame, size=640)
#     list_plates = plates.pandas().xyxy[0].values.tolist()
#     list_read_plates = set()
    
#     for cc in range(2):
#         for ct in range(2):
#             lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
#             if lp != "unknown":
#                 list_read_plates.add(lp)

#                 # Font size and thickness
#                 font_scale = 1.5  # Tăng kích thước chữ
#                 thickness = 4      # Tăng độ đậm
#                 text_color = (255, 255, 255)  # Chữ trắng
#                 outline_color = (0, 0, 0)  # Viền đen
#                 text_position = (x, y - 10)

#                 # Vẽ viền đen (viết chữ với độ dày lớn hơn)
#                 cv2.putText(frame, lp, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
#                             font_scale, outline_color, thickness + 2, cv2.LINE_AA)

#                 # Viết chữ trắng
#                 cv2.putText(frame, lp, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
#                             font_scale, text_color, thickness, cv2.LINE_AA)

#                 flag = 1
#                 break
#         if flag == 1:
#             break

#     # Calculate FPS
#     new_frame_time = time.time()
#     fps = int(1 / (new_frame_time - prev_frame_time))
#     prev_frame_time = new_frame_time

#     # Overlay FPS on the frame
#     cv2.putText(frame, f"FPS: {fps}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

#     # Show frame
#     cv2.imshow('frame', frame)

#     # Write frame to output video
#     video_writer.write(frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# vid.release()
# video_writer.release()
# cv2.destroyAllWindows()

# print("Video saved as output.mp4")

from PIL import Image
import cv2
import torch
import math 
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import time
import argparse
import function.helper as helper

# Load YOLO models
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

prev_frame_time = 0
new_frame_time = 0

# Open video file
vid = cv2.VideoCapture('/Users/kaiser_1/Documents/Zalo Received Files/QNO-6082R_20250325073324.avi')

# Get video properties
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
fps = int(vid.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
video_writer = cv2.VideoWriter('/Users/kaiser_1/Documents/GitHub/License-Plate-Recognition/output_mp4/output2.mp4', fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = vid.read()
    if not ret:
        break  # Exit if video ends
    
    plates = yolo_LP_detect(frame, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()
    
    for plate in list_plates:
        flag = 0
        x = int(plate[0])  # xmin
        y = int(plate[1])  # ymin
        w = int(plate[2] - plate[0])  # xmax - xmin
        h = int(plate[3] - plate[1])  # ymax - ymin  
        crop_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 225), thickness=2)
        
        cv2.imwrite("crop.jpg", crop_img)
        rc_image = cv2.imread("crop.jpg")
        lp = ""

        for cc in range(2):
            for ct in range(2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)

                    # Font settings
                    font_scale = 2.5  # Increase font size
                    thickness = 6      # Increase text thickness
                    text_color = (255, 255, 255)  # White text
                    outline_color = (0, 0, 0)  # Black outline
                    text_position = (x, y - 10)

                    # Draw black outline (thicker text)
                    cv2.putText(frame, lp, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                                font_scale, outline_color, thickness + 2, cv2.LINE_AA)

                    # Draw white text
                    cv2.putText(frame, lp, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                                font_scale, text_color, thickness, cv2.LINE_AA)
                    
                    flag = 1
                    break
            if flag == 1:
                break

    # Calculate FPS
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    prev_frame_time = new_frame_time

    # Overlay FPS on the frame
    cv2.putText(frame, f"FPS: {fps}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # Show frame
    cv2.imshow('frame', frame)

    # Write frame to output video
    video_writer.write(frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
vid.release()
video_writer.release()
cv2.destroyAllWindows()

print("Video saved as output.mp4")