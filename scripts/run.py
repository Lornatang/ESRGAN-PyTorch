# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

# Prepare dataset
os.system("python3 ./prepare_dataset.py --images_dir ../data/DIV2K/original/DIV2K_train_HR --output_dir ../data/DIV2K/ESRGAN/train --image_size 544 --step 272 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/DIV2K/original/DIV2K_valid_HR --output_dir ../data/DIV2K/ESRGAN/valid --image_size 544 --step 544 --num_workers 16")
