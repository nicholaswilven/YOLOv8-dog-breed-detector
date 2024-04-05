from ultralytics import YOLO

def train_model(data_config : str = "stanford_dog_dataset_v1.yaml"):
    yolo = YOLO("yolov8n.pt")
    yolo.train(data = data_config, epochs = 10)