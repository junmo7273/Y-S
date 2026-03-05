import torch
import datetime
import os
from pathlib import Path
from ultralytics import YOLO
import albumentations as A  


if __name__ == '__main__':
    try:
        # Compatibility handling: use official YOLOv11 automatically if custom_yolov11 is not available
        try:
            from custom_yolov11 import CustomYOLOv11
            custom_model_available = True
        except ImportError:
            print("CustomYOLOv11 not found, using official YOLOv11 instead")
            custom_model_available = False

        # -------------------------- 1. Path and Parameter Configuration --------------------------
        model_path = "./weights/yolo_nano_solar_panel.pt"
        output_root = Path("./runs/detect/train/weights")
        output_root.mkdir(parents=True, exist_ok=True)
        device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # -------------------------- 2. Load Model (Compatible with Custom/Official) --------------------------
        if custom_model_available:
            model = CustomYOLOv11(cfg='yolov11n.yaml', nc=7)  # Replace with actual number of classes
            if os.path.exists(model_path):
                model.load(model_path)  # Load pretrained weights
                print(f"Loaded pretrained weights from {model_path}")
        else:
            # Official YOLOv11 loading method (fallback solution)
            if os.path.exists(model_path):
                model = YOLO(model_path)
                print(f"Loaded pretrained weights from {model_path}")
            else:
                model = YOLO('yolov11n.yaml')
                print("Using official YOLOv11n default weights")

        # -------------------------- 3. Model Training (Pure parameter configuration augmentation, no import dependencies) --------------------------
        train_results = model.train(
            data="./dataset/data.yaml",
            epochs=100,
            imgsz=640,
            device=device,
            batch=16,
            save=True,
            project=str(output_root.parent.parent),
            name="train_enhanced",
            exist_ok=True,
            patience=50,
            pretrained=True,
            
            # ===================== PV Panel-Specific Augmentation Parameters (Replace manual Compose) =====================
            # 1. Basic augmentation 
            hsv_h=0.015,  # Hue augmentation 
            hsv_s=0.7,    # Saturation augmentation 
            hsv_v=0.4,    # Value (brightness) augmentation 
            degrees=15.0, # Rotation angle 
            translate=0.1, # Translation 
            scale=0.1,    # Scaling 
            perspective=0.001, # Perspective transformation 
            flipud=0.0,   # Probability of vertical flip 
            fliplr=0.5,   # Probability of horizontal flip 
            mosaic=1.0,   # Mosaic augmentation 
            mixup=0.1,    # MixUp augmentation
            copy_paste=0.0, # Copy-paste augmentation 
            
            # 2. Small target detection optimization
            val=True,
            plots=True,
            rect=False,  # Disable rectangular training to improve small target detection
            single_cls=False,  # Keep False for multi-class detection
            # Enable built-in augmentation (core: no need to manually import any augmentation classes)
            augment=True,
        )

        # -------------------------- 4. Validation and Export (Keep original logic) --------------------------
        val_results = model.val()
        print(f"\nValidation results summary: {val_results.results_dict}")

        # Export ONNX
        try:
            onnx_export_path = model.export(
                format="onnx",
                opset=12,
                half=device != 'cpu',
                simplify=True,
                imgsz=640,
            )
            print(f"\nONNX model exported successfully: {onnx_export_path}")
        except Exception as e:
            print(f"\nONNX export error: {e}")

        # Save PT model
        now = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        pt_save_path = output_root / f"yolov11n_solar_enhanced_{now}.pt"
        model.save(str(pt_save_path))
        print(f"\nEnhanced model saved to: {pt_save_path}")

        # Export TorchScript
        try:
            ts_export_path = model.export(
                format="torchscript",
                imgsz=640,
                half=device != 'cpu',
            )
            print(f"\nTorchScript model exported: {ts_export_path}")
        except Exception as e:
            print(f"\nTorchScript export error: {e}")
            
    except ImportError as e:
        print(f"Dependency error: {e}\nPlease install required packages: pip install ultralytics torch albumentations")
    except Exception as e:
        print(f"\nRuntime error: {e}")