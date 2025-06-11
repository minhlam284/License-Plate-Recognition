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

# 👉 Sử dụng webcam làm nguồn video (0 là camera mặc định)
vid = cv2.VideoCapture(0)

# Nếu cần lưu lại video đầu ra
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
fps = 30  # Có thể chỉnh tay nếu camera không cung cấp fps chính xác

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_writer = cv2.VideoWriter('output_realtime.mp4', fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = vid.read()
    if not ret:
        print("Không nhận được khung hình từ camera.")
        break
    
    plates = yolo_LP_detect(frame, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()
    
    for plate in list_plates:
        flag = 0
        x = int(plate[0])
        y = int(plate[1])
        w = int(plate[2] - plate[0])
        h = int(plate[3] - plate[1])
        crop_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 225), thickness=2)

        lp = ""
        for cc in range(2):
            for ct in range(2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)

                    font_scale = 2.5
                    thickness = 6
                    text_color = (255, 255, 255)
                    outline_color = (0, 0, 0)
                    text_position = (x, y - 10)

                    # Outline text
                    cv2.putText(frame, lp, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, outline_color, thickness + 2, cv2.LINE_AA)
                    # Main text
                    cv2.putText(frame, lp, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, text_color, thickness, cv2.LINE_AA)
                    
                    flag = 1
                    break
            if flag == 1:
                break

    # FPS calculation
    new_frame_time = time.time()
    if prev_frame_time != 0:
        fps = int(1 / (new_frame_time - prev_frame_time))
    prev_frame_time = new_frame_time

    cv2.putText(frame, f"FPS: {fps}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)

    # Display
    cv2.imshow('Real-time License Plate Detection', frame)

    # Optional: write to file
    # video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
vid.release()
# video_writer.release()
cv2.destroyAllWindows()

print("Real-time processing ended. Output saved.")
