# webcam_enhanced.py
import cv2
import torch
import time
from ultralytics import YOLO
import numpy as np

# -------------------------- 1. Load Enhanced Model --------------------------
model_path = "./weights/yolov11n_solar_enhanced_xxxxxx.pt"  # Replace with the path of trained model
try:
    # Load custom model (automatically adapt to C2f-Lite and small target branch)
    model = YOLO(model_path)
    print(f"Enhanced model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# -------------------------- 2. Camera Configuration --------------------------
cap = cv2.VideoCapture(0)
# Improve camera resolution (beneficial for small target detection)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {frame_width}x{frame_height}")

# -------------------------- 3. Small Target Detection Optimization Parameters --------------------------
CONF_THRESHOLD = 0.2  # Lower confidence threshold (adapt to small targets)
IOU_THRESHOLD = 0.45   # Adjust IOU threshold
IMG_SIZE = 640         # Inference size (consistent with training)

# -------------------------- 4. FPS Calculation and Window Configuration --------------------------
prev_frame_time = 0
new_frame_time = 0
cv2.namedWindow("YOLOv11 Solar Panel Detection (Small Target Enhanced)", cv2.WINDOW_NORMAL)

# -------------------------- 5. Small Target Detection Preprocessing --------------------------
def preprocess_small_target(frame):
    """Small target preprocessing: enhance contrast and zoom local areas"""
    # 1. Local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe_gray = clahe.apply(gray)
    frame = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
    
    # 2. Slight zoom (highlight small targets)
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (int(w*1.2), int(h*1.2)), interpolation=cv2.INTER_LINEAR)
    
    # 3. Limit size (adapt to model input)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return frame

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # -------------------------- 6. Small Target Preprocessing and Inference --------------------------
        # Backup original frame (for saving screenshots)
        frame_original = frame.copy()
        
        # Small target preprocessing
        frame_processed = preprocess_small_target(frame)
        
        # Inference (optimize small target detection parameters)
        results = model(
            frame_processed, 
            conf=CONF_THRESHOLD, 
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            augment=True,  # Enable mild augmentation during inference
            retina_masks=True  # High-resolution detection
        )

        # -------------------------- 7. Result Visualization --------------------------
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {fps:.1f}"

        # Draw detection results (restore to original resolution)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
        
        # Add FPS and small target detection identifier
        cv2.putText(annotated_frame, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Small Target Enhanced", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Count small targets
        small_target_count = 0
        if hasattr(results[0], 'boxes') and hasattr(results[0].boxes, 'cls'):
            for box in results[0].boxes:
                # Determine small targets: area < 0.1% of total pixels
                x1, y1, x2, y2 = box.xyxy[0]
                area = (x2 - x1) * (y2 - y1)
                if area < (IMG_SIZE*IMG_SIZE)*0.001:
                    small_target_count += 1
            cv2.putText(annotated_frame, f"Small Targets: {small_target_count}", 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display results
        cv2.imshow("YOLOv11 Solar Panel Detection (Small Target Enhanced)", annotated_frame)

        # -------------------------- 8. Key Control --------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_name = f"detection_small_target_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_name, frame_original)
            print(f"Screenshot saved as {screenshot_name}")

except KeyboardInterrupt:
    print("Detection stopped by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released")