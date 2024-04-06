YOLOv8 Dog Breed Detector
=========================

Welcome to the YOLOv8 Dog Breed Detector repository! This repository contains tools and scripts for finetuning the YOLOv8 model on the Stanford Dog Dataset, enabling you to detect dog breeds in images and videos.

Features
--------

*   **Automatic Dataset Download**: Easily download the Stanford Dog Dataset using Python subprocess and Linux commands.
*   **Dataset Formatting**: Format the downloaded dataset into the Ultralytics YOLOv8 dataset format and generate the required `.yaml` file.
*   **Model Training**: Train the YOLOv8 model on the formatted dataset using CPU or GPU resources.
*   **Video Inference**: Perform object detection on videos using the trained model and generate annotated video files.
*   **Asynchronous Functions**: Utilize asynchronous functions for improved efficiency and parallel execution.

Usage
-----

1.  **Clone the Repository**:
    `git clone https://github.com/your_username/yolov8_dog_breed_detector.git`
    
2.  **Download Dataset**:
    `python main.py -dd`
    
3.  **Format Dataset**:
    `python main.py -pd`
    
4.  **Train Model**:
    `python main.py -tm`
    
5.  **Infer Videos**:
    `python main.py -im`
    

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
----------------

*   [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - For providing the YOLOv8 model implementation.
*   [Stanford Dog Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) - For providing the dog breed dataset.
