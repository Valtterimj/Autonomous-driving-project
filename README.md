

## Set up

### 1. clone repo

git clone https://github.com/Valtterimj/Autonomous-driving-project.git

cd autonomous-driving-project

### 2. set up environment 
#### using uv
uv sync  
uv pip install -e .

#### using pip
python -m vevn .venv  
source .venv/bin/activate  
pip install -e .

### 3. laod the kitti data
download left color images of object data set (12GB) and traiing labels of object data set from
https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d

add the files to
data/kitti/raw/  
    image_2/  
    label_2/  

## run pipeline
runs the main function  
python -m kitti_object_detection.main  