# yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
# https://github.com/ultralytics/ultralytics

from ultralytics import YOLO
import torch
import os

def train_save_cls():
    # Load a model
    model = YOLO("yolo11n.pt")

    # Train the model
    train_results = model.train(
        data="coco8.yaml",  # path to dataset YAML
        epochs=5,  # number of training epochs
        imgsz=640,  # training image size
        device="mps",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    os.makedirs("model", exist_ok=True)

    torch.save(model, "model/yolo_cls.pth")

def predict_cls(model_path, image_path):

    model = torch.load(model_path)
    # Perform object detection on an image
    results = model(f"cam_images/{image_path}")
    results[0].show()

if __name__ == "__main__":
    # train_save_cls()
    # predict_cls("model/yolo_cls.pth")
    # Load YOLO11n-pose, train it on COCO8-pose for 3 epochs and predict an image with it
    from ultralytics import YOLO

    model = YOLO('./yolo11n-pose.pt')  # load a pretrained YOLO pose model
    # model.train(data='coco8-pose.yaml', epochs=3)  # train the model
    # Export the model
    # model.export()
    results = model("cam_images/captured")
    results[0].show()