import os
import cv2
import torch
import random
import function.utils_rotate as utils_rotate
import function.helper as helper
from tqdm import tqdm

# Định nghĩa thư mục đầu vào và đầu ra
input_folder = "/Users/kaiser_1/Downloads/Anh bien số/13"
output_with_plate = "/Users/kaiser_1/Documents/GitHub/License-Plate-Recognition/output_with_plate1"
output_without_plate = "/Users/kaiser_1/Documents/GitHub/License-Plate-Recognition/output_without_plate1"

os.makedirs(output_with_plate, exist_ok=True)
os.makedirs(output_without_plate, exist_ok=True)

# Load mô hình YOLOv5
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.30

# Lấy danh sách ảnh trong thư mục đầu vào
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(image_files)
selected_images = image_files  # Chọn 100 ảnh ngẫu nhiên

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
    df_plates = plates.pandas().xyxy[0]
    list_plates = df_plates.values.tolist()
    has_plate = False
    # conf_file_path = os.path.join(output_with_plate, img_name.rsplit('.',1)[0] + '_conf.txt')
    # with open(conf_file_path, 'w') as conf_file:
    if len(list_plates) == 0:
        lp = helper.read_plate(yolo_license_plate, img)
        if lp != "unknown":
            font_scale = max(0.5, min(img.shape[1] / 800, img.shape[0] / 600))
            cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (36, 255, 12), 2)
            has_plate = True
            # conf_file.write(f"{lp} confidence: -1\n")
    else:
        for plate in list_plates:
            x1, y1, x2, y2, conf, cls, name = plate
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            crop_img = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # conf_file.write(f"{name} confidence: {conf:.4f}\n")

            flag = 0
            for cc in range(2):
                for ct in range(2):
                    processed_img = utils_rotate.deskew(crop_img, cc, ct)
                    lp = helper.read_plate(yolo_license_plate, processed_img)
                    # preds = yolo_license_plate(processed_img, size=320)
                    # df = preds.pandas().xyxy[0]
                    # lp_chars = []
                    # confs = []

                    # for idx, row in df.iterrows():
                    #     char = row['name']
                    #     conf_char = row['confidence']
                    #     lp_chars.append(char)
                    #     confs.append(conf_char)
                    if lp != "unknown":
                        font_scale = max(0.5, min(w / 200, h / 100))
                        # lp_text = ''.join(lp_chars)
                        # avg_conf = sum(confs) / len(confs)
                        # cv2.putText(img, lp, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (36, 255, 12), 2)
                        cv2.putText(img, f"{lp} ({conf:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (36, 255, 12), 2)
                        has_plate = True
                        flag = 1
                        break
                if flag:
                    break

    # Xác định thư mục đầu ra dựa trên trạng thái `has_plate`
    output_path = os.path.join(output_with_plate if has_plate else output_without_plate, img_name)
    cv2.imwrite(output_path, img)

print(f"Quá trình infer hoàn tất. Kết quả đã lưu vào: \n- {output_with_plate}\n- {output_without_plate}")