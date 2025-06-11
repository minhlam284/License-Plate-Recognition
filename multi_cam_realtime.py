import cv2
import torch
import math
import time
import threading
import function.utils_rotate as utils_rotate
import function.helper as helper

# Load YOLO models
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

def process_camera(cam_id, window_name):
    cap = cv2.VideoCapture(cam_id)
    prev_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Không nhận được khung hình từ camera {cam_id}")
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

                        # Outline + main text
                        cv2.putText(frame, lp, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + 2, cv2.LINE_AA)
                        cv2.putText(frame, lp, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

                        flag = 1
                        break
                if flag == 1:
                    break

        # FPS calculation
        new_frame_time = time.time()
        if prev_frame_time != 0:
            fps = int(1 / (new_frame_time - prev_frame_time))
            cv2.putText(frame, f"FPS: {fps}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)
        prev_frame_time = new_frame_time

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)
    print(f"Camera {cam_id} stopped.")

# Khởi động 2 camera ở 2 thread khác nhau
thread1 = threading.Thread(target=process_camera, args=(0, "Camera Vào"))
thread2 = threading.Thread(target=process_camera, args=(1, "Camera Ra"))

thread1.start()
thread2.start()

thread1.join()
thread2.join()

cv2.destroyAllWindows()
print("Kết thúc chương trình.")
