import os

# Prepare dataset
os.system("python ./prepare_dataset.py --images_dir ../data/DFO2K/original/train --output_dir ../data/DFO2K/ESRGAN/train --image_size 204 --step 102")
os.system("python ./prepare_dataset.py --images_dir ../data/DFO2K/original/valid --output_dir ../data/DFO2K/ESRGAN/train --image_size 204 --step 102")

# Create LMDB database file
os.system("python ./create_lmdb_dataset.py --images_dir ../data/DFO2K/ESRGAN/train --lmdb_path ../data/train_lmdb/ESRGAN/DFO2K_HR_lmdb --upscale_factor 1")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/DFO2K/ESRGAN/train --lmdb_path ../data/train_lmdb/ESRGAN/DFO2K_LRbicx4_lmdb --upscale_factor 4")

os.system("python ./create_lmdb_dataset.py --images_dir ../data/DFO2K/ESRGAN/valid --lmdb_path ../data/valid_lmdb/ESRGAN/DFO2K_HR_lmdb --upscale_factor 1")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/DFO2K/ESRGAN/valid --lmdb_path ../data/valid_lmdb/ESRGAN/DFO2K_LRbicx4_lmdb --upscale_factor 4")
