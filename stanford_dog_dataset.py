import os
import asyncio
from aiofiles import open as async_open
from mat4py import loadmat
from itertools import chain
import xml.etree.ElementTree as ET
import re
import yaml
import pandas as pd

from utils import tqdm_async, run_cmd_from, async_timer

data_folder_name = "stanford_dog_dataset"
dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs"
raw_dataset_path = f"data/{data_folder_name}/raw"

async def download_and_unzip_file(file_url, target_dir):
    # Step 1: Download file
    cmd_download = f"wget {file_url} -P {target_dir}"
    await run_cmd_from(cmd_download)
    print(f'Downloaded {file_url}')
    
    # Step 2: Unzip file
    filename = os.path.basename(file_url)
    if filename.endswith(".tar"):
        cmd_unzip = f"tar -xvf {filename}"
        await run_cmd_from(cmd_unzip, target_dir)
        print(f'Unzipped {target_dir}/{filename}')
        
        # Step 3: Delete downloaded file
        cmd_delete = f"rm {filename}"
        await run_cmd_from(cmd_delete, target_dir)
        print(f'Deleted {target_dir}/{filename}')

@async_timer
async def download_and_unzip_dataset():
    print(f'Downloading Stanford Dog Dataset from {dataset_url}')
    # Create directory if not exist
    os.makedirs(raw_dataset_path, exist_ok=True)
    files_to_download = [f"{dataset_url}/{filename}" for filename in ['images.tar', 'annotation.tar', 'lists.tar']]
    # Download and unzip files asynchronously
    await asyncio.gather(*[download_and_unzip_file(file_url, raw_dataset_path) for file_url in files_to_download])
    print(f'Downloaded Stanford Dog Dataset from {dataset_url}!')

def parse_xml_annot_for_YOLOv8(filename : str, label_classes : list):
    # Parse XML annotation file to readable YOLOv8 TXT annotation file
    tree = ET.parse(filename)
    root = tree.getroot()

    # Extract image size
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # Extract bounding box information
    bndbox = []
    objects = root.findall('object')
    for i, obj in enumerate(objects):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        xcenter = (xmax+xmin)/(2*width)
        ycenter = (ymax+ymin)/(2*height)
        xwidth = (xmax-xmin)/(width)
        yheight = (ymax-ymin)/(height)
        bndbox.append(f"{label_classes[i]} {xcenter} {ycenter} {xwidth} {yheight}")
        return "\n".join(bndbox)

async def copy_file(src_path: str, dest_path: str):
    cmd_copy = f"cp {src_path} {dest_path}"
    await run_cmd_from(cmd_copy)

async def parse_and_copy_files(img_path, annot_path, label_classes, target_dir):
    # Complete parsing process for one file
    annot_content = parse_xml_annot_for_YOLOv8(os.path.join(raw_dataset_path, "Annotation", annot_path), label_classes)
    target_annot_path = os.path.join(target_dir, os.path.basename(annot_path)+".txt")
    async with async_open(target_annot_path, 'w') as file:
        await file.write(annot_content)
    await copy_file(os.path.join(raw_dataset_path, "Images", img_path), target_dir)

@tqdm_async(total=20580, desc="Progress")
async def parse_dataset(idx: str = 'v1'):
    target_dir = f"data/{data_folder_name}/{idx}"
    os.makedirs(os.path.join(target_dir, "Training"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "Testing"), exist_ok=True)

    print("Processing Training split")
    training_metadata = loadmat(os.path.join(raw_dataset_path, 'train_list.mat'))
    await asyncio.gather(*[parse_and_copy_files(img_path, annot_path, label_classes, os.path.join(target_dir, "Training")) for img_path, annot_path, label_classes in zip(chain.from_iterable(training_metadata["file_list"]), chain.from_iterable(training_metadata["annotation_list"]), training_metadata["labels"])])

    print("Processing Testing split")
    testing_metadata = loadmat(os.path.join(raw_dataset_path, 'test_list.mat'))
    await asyncio.gather(*[parse_and_copy_files(img_path, annot_path, label_classes, os.path.join(target_dir, "Testing")) for img_path, annot_path, label_classes in zip(chain.from_iterable(testing_metadata["file_list"]), chain.from_iterable(testing_metadata["annotation_list"]), testing_metadata["labels"])])

# Example usage:
async def main():
    if not os.path.exists(raw_dataset_path):
        await download_and_unzip_dataset()
    await parse_dataset()

asyncio.run(main())

def get_dog_breed_name(filepath):
        match = re.search(r"-(.+?)/", filepath)
        if match:
            dog_breed_name = match.group(1)
            return dog_breed_name.replace("_"," ").title()

def generate_class_labeling():
    files_metadata = loadmat(os.path.join(raw_dataset_path, 'file_list.mat'))
    df = pd.DataFrame(files_metadata)
    for col in df.columns:
        df[col] = df[col].apply(lambda x:x[0])
    return df.drop_duplicates("labels").set_index("labels")['file_list'].apply(get_dog_breed_name).to_dict()

def generate_yaml(idx: str = 'v1'):
    target_dir = f"data/{data_folder_name}/{idx}"
    class_labeling = generate_class_labeling()
    data = {
        'train': target_dir + '/Training',
        'test': target_dir + '/Testing',
        'nc': len(class_labeling),
        'names': class_labeling
    }

    filename = f"{data_folder_name}_{idx}.yaml"
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style = False)

