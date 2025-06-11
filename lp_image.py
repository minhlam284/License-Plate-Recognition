import cv2
import torch
import function.utils_rotate as utils_rotate
import function.helper as helper
import argparse

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = ap.parse_args()

# Load YOLOv5 models
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.30

# Read image
img = cv2.imread(args.image)
height, width, _ = img.shape
font_scale = min(width, height) / 1000  # Adjust text size based on image size
thickness = max(1, int(font_scale * 2))

# Detect license plates
plates = yolo_LP_detect(img, size=640)
list_plates = plates.pandas().xyxy[0].values.tolist()
list_read_plates = set()

if len(list_plates) == 0:
    lp = helper.read_plate(yolo_license_plate, img)
    if lp != "unknown":
        cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (36, 255, 12), thickness)
        list_read_plates.add(lp)
else:
    for plate in list_plates:
        flag = 0
        x, y, w, h = int(plate[0]), int(plate[1]), int(plate[2] - plate[0]), int(plate[3] - plate[1])
        crop_img = img[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness)
        
        for cc in range(2):
            for ct in range(2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)
                    cv2.putText(img, lp, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (36, 255, 12), thickness)
                    flag = 1
                    break
            if flag:
                break

cv2.imshow('frame', img)
cv2.waitKey()
cv2.destroyAllWindows()