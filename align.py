import cv2
import numpy as np

def align_license_plate(image_path, output_path, bbox=None):
    """
    Args:
    - image_path: Đường dẫn đến ảnh gốc.
    - output_path: Đường dẫn lưu ảnh căn chỉnh.
    - bbox: (Optional) Tọa độ bbox dạng (x, y, w, h)
    """
    image = cv2.imread(image_path)

    if bbox is None:
        print("No bbox provided. Detecting license plate region...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 2 <= aspect_ratio <= 5:  # Giả sử biển số có tỷ lệ 2:1 đến 5:1
                bbox = (x, y, w, h)
                break
        if bbox is None:
            print("Failed to detect license plate region.")
            return

    x, y, w, h = bbox
    width, height = w, h
    bbox_points = np.array([
        [x, y],          # Top-left
        [x + w, y],      # Top-right
        [x + w, y + h],  # Bottom-right
        [x, y + h],      # Bottom-left
    ], dtype="float32")

    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(bbox_points, dst_points)
    aligned_plate = cv2.warpPerspective(image, M, (width, height))

    cv2.imwrite(output_path, aligned_plate)
    print(f"Aligned license plate saved to {output_path}")

image_path = "car_image.jpg"
output_path = "aligned_plate_with_bbox.jpg"
bbox = [(50, 50), (200, 200)]
align_license_plate(image_path, output_path, bbox)

output_path_no_bbox = "aligned_plate_no_bbox.jpg"
align_license_plate(image_path, output_path_no_bbox)