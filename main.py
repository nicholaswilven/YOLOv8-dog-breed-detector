import asyncio
import argparse

from stanford_dog_dataset import download_and_unzip_dataset, parse_dataset, generate_yaml
from trainer import train_model
parser = argparse.ArgumentParser(description='YOLOv8 for Dog Breed Detector')

parser.add_argument("--download_data", "-dd", action="store_true", help="Download Stanford Dog Dataset")
parser.add_argument("--preprocess_data", "-pd", action="store_true", help="Preprocess data into YOLOv8 format")
parser.add_argument("--generate_yaml", "-gy", action="store_true", help="Write .yaml for indexed data")
parser.add_argument("--train_model", "-tm", action="store_true", help="Train YOLOv8 nano model using specified config")

args = parser.parse_args()

async def main():
    args_download_data = args.download_data
    args_preprocess_data = args.preprocess_data
    args_generate_yaml = args.generate_yaml
    args_train_model = args.train_model

    if args_download_data:
        await download_and_unzip_dataset()
    if args_preprocess_data:
        await parse_dataset()
    if args_generate_yaml:
        await generate_yaml()
    if args_train_model:
        await train_model()

asyncio.run(main())