import os

# Prepare dataset
os.system("python ./prepare_dataset.py --images_dir ../data/DFO2K/original/train --output_dir ../data/DFO2K/ESRGAN/train --image_size 204 --step 102 --num_workers 10")
os.system("python ./prepare_dataset.py --images_dir ../data/DFO2K/original/valid --output_dir ../data/DFO2K/ESRGAN/valid --image_size 204 --step 102 --num_workers 10")
