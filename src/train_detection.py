
from ultralytics import YOLO
import torch

def main():
    print("GPU :", torch.cuda.get_device_name(0))

    # Charger le modèle pré-entraîné
    model = YOLO("yolov8n.pt")

    # Entraînement
    model.train(
        data="dataset/detection_yolo/data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        patience=20,
        project="runs/detection",
        name="candy_yolov8n"
    )

if __name__ == "__main__":
    main()
