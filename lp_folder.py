import os
import cv2
import torch
import random
import function.utils_rotate as utils_rotate
import function.helper as helper
from tqdm import tqdm

# Định nghĩa thư mục đầu vào và đầu ra
input_folder = "/Users/kaiser_1/Downloads/Anh bien số/13"
output_folder = "/Users/kaiser_1/Documents/GitHub/License-Plate-Recognition/output"
os.makedirs(output_folder, exist_ok=True)

# Load mô hình YOLOv5
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.30

# Lấy danh sách ảnh trong thư mục đầu vào
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(image_files)
selected_images = image_files[:100]  # Chọn 100 ảnh ngẫu nhiên

for img_name in tqdm(selected_images, desc="Processing images"):
    img_path = os.path.join(input_folder, img_name)

    if not os.path.isfile(img_path):
        print(f"File không tồn tại: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"Lỗi khi đọc ảnh: {img_path}")
        continue
    
    plates = yolo_LP_detect(img, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()

    if len(list_plates) == 0:
        lp = helper.read_plate(yolo_license_plate, img)
        if lp != "unknown":
            font_scale = max(0.5, min(img.shape[1] / 800, img.shape[0] / 600))
            cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (36, 255, 12), 2)
            list_read_plates.add(lp)
    else:
        for plate in list_plates:
            flag = 0
            x, y, w, h = int(plate[0]), int(plate[1]), int(plate[2] - plate[0]), int(plate[3] - plate[1])
            crop_img = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            for cc in range(2):
                for ct in range(2):
                    processed_img = utils_rotate.deskew(crop_img, cc, ct)
                    lp = helper.read_plate(yolo_license_plate, processed_img)
                    if lp != "unknown":
                        list_read_plates.add(lp)
                        font_scale = max(0.5, min(w / 200, h / 100))
                        cv2.putText(img, lp, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (36, 255, 12), 2)
                        flag = 1
                        break
                if flag:
                    break

    output_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_path, img)
