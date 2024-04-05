import os
import subprocess

def run_cmd_from(cmd : str, path : str = '.'):
    result = subprocess.run(f"cd {path} && {cmd}", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, universal_newlines=True)

data_folder_name = "stanford_dog_dataset"
dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs"
raw_dataset_path = f"data/{data_folder_name}/raw"

def download_dataset():
    print(f'Downloading Stanford Dog Dataset from {dataset_url}')
    
    # Download and unzip dataset from official website using linux commands
    os.makedirs(raw_dataset_path, exist_ok=True)

    files_to_download = ['images.tar', 'annotation.tar', 'lists.tar']

    for file in files_to_download:
        # Step 1: Download images.tar and annotation.tar
        cmd_download = f"wget {dataset_url}/{file}"
        result = run_cmd_from(cmd_download, raw_dataset_path)
        print(f'Downloaded {file}')
        # Step 2: Unzip images.tar and annotation.tar
        cmd_unzip = f"tar -xvf {file}"
        result = run_cmd_from(cmd_unzip, raw_dataset_path)
        print(f'Unzipped {file}')
        # Step 3: Delete unused images.tar and annotation.tar
        cmd_delete = f"rm {file}"
        result = run_cmd_from(cmd_delete, raw_dataset_path)
        print(f'Deleted {file}')
    print(f'Downloaded Stanford Dog Dataset from {dataset_url}!')

def explode_dataset():
    # NOT NEEDED -> NOT USED
    pass
    # Explode Images and Annocation for each class into one folder only for YOLOv8
    print(f'Exploding Stanford Dog Dataset on {raw_dataset_path}')
    subfolders = [f.path for f in os.scandir(raw_dataset_path + "/Images") if f.is_dir()] + [f.path for f in os.scandir(raw_dataset_path+ "/Annotation") if f.is_dir()]

    for subfolder in subfolders:
        cmd_move = f"mv {subfolder}/* {raw_dataset_path}"
        cmd_remove = f"rmdir {subfolder}"
        run_cmd_from(cmd_move, raw_dataset_path)
        run_cmd_from(cmd_remove, raw_dataset_path)

    run_cmd_from("rmdir Images")
    run_cmd_from("rmdir Annotation")
    print(f'Exploded Stanford Dog Dataset on {raw_dataset_path}!')

import xml.etree.ElementTree as ET
def parse_xml_annot_for_YOLOv8(filename : str, label_class : int):
    # Parse XML annotation file to readable YOLOv8 TXT annotation file
    tree = ET.parse(filename)
    root = tree.getroot()

    # Extract image size
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # Extract bounding box information
    obj = root.find('object')
    bbox = obj.find('bndbox')
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
    xcenter = (xmax+xmin)/(2*width)
    ycenter = (ymax+ymin)/(2*height)
    xwidth = (xmax-xmin)/(width)
    yheight = (ymax-ymin)/(height)
    bndbox = [label_class, xcenter, ycenter, xwidth, yheight]
    return " ".join(bndbox)

from mat4py import loadmat
def parse_dataset(idx : str = 'v1'):
    target_dir = f"data/{data_folder_name}/{idx}"

    os.makedirs(target_dir + "/Training", exist_ok=True)
    os.makedirs(target_dir + "/Testing", exist_ok=True)
    # os.makedirs(target_dir + "/Validation", exist_ok=True)

    print("Processing Training split")
    training_metadata = loadmat(raw_dataset_path + '/train_list.mat')
    
    for img_path, annot_path, label_class in zip(training_metadata["file_list"], training_metadata["annotation_list"], training_metadata["labels"]):
        # Parse the annotation
        annot_content = parse_xml_annot_for_YOLOv8(raw_dataset_path + "/Annotation/" + annot_path[0], label_class)
        target_annot_path = target_dir + "/Training/" + annot_path[0].split("/")[-1]
        with open(target_annot_path, 'w') as file:
            file.write(annot_content)
        # Copy the jpg to File
        cmd_move = f"cp {raw_dataset_path}/Images/{img_path}.jpg {target_dir}/Training"
        run_cmd_from(cmd_move)

    print("Processing Testing split")
    testing_metadata = loadmat(raw_dataset_path + '/test_list.mat') 
    for img_path, annot_path, label_class in zip(testing_metadata["file_list"], testing_metadata["annotation_list"], testing_metadata["labels"]):
        # Parse the annotation
        annot_content = parse_xml_annot_for_YOLOv8(raw_dataset_path + "/Annotation/" + annot_path[0], label_class)
        target_annot_path = target_dir + "/Testing/" + annot_path[0].split("/")[-1]
        with open(target_annot_path, 'w') as file:
            file.write(annot_content)
        # Copy the jpg file
        cmd_move = f"cp {raw_dataset_path}/Images/{img_path}.jpg {target_dir}/Testing"
        run_cmd_from(cmd_move)