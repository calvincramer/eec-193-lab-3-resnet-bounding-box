# Train YOLOv3 and Mask-RCNN with COCO database

### Install docker
Install docker on your computer. Depending on your operating system, follow the given guides:
* Linux ->
* Mac ->
* Windows ->

### Get Ubuntu 16.0 docker image
Get the docker image for the latest version of ubuntu using:
'docker pull ubuntu:16.04'
Then make sure the image was pulled correctly by using:
'docker images'
And you should see the 'ubuntu' image with the tab '16.04' in the list.


### Make container, start and attach
Make the ubuntu docker container using:
'docker container create -it --name <container_name> ubuntu /bin/bash'
Then if you type:
'docker ps -a'
You should see '<container_name>' in the list of all containers
Then start and attach to the container:
'docker start <container_name>'
'docker attach <container_name>'
You should now be inside the running docker container.
If you don't see a shell prompt, press ctrl+c.

### Install git
First of all, update the libraries that came within the container:
'apt-get update'
Now install git:
'apt-get install git

### Install curl
We need this to install anaconda
'apt-get install curl'

### Install anaconda
Follow this [guide](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04)
You can stop after the 'conda list' step.


## YOLOv3

........................ DONT NEED THIS
### Install dependencies
* Install numpy:
'conda install numpy'
Although numpy comes with conda, this will update numpy to the latest version

* Install torch
First install cmake:
'apt-get install cmake'
and build-essential:
'apt-get install build-essential'
Now clone the torch repository:
'git clone http://github.com/torch/distro.git ~/torch --recursive'
git clone ultralytics/yolov3
..........................




We will be using eriklindernoren's implementation of YOLOv3:
Clone the git repository
'git clone https://github.com/eriklindernoren/PyTorch-YOLOv3'
Were going to use pip to install numpy, torch, torchvision, pillow, and matplotlib
'apt-get install python3-pip'
Navigate into the repository we just clones
'cd PyTorch-YOLOv3'
Install the dependencies needed for PyTorch-YOLOv3
'pip3 install -r requirements.txt'

If you want to use the YOLO model with pretrained weights, follow these steps:
'apt-get install wget'
'cd weights'
'bash download_weights.sh'


#### Now dowload the COCO dataset:
Make sure you are in the PyTorch-YOLOv3 base directory. If you are already in the weights folder just do
'cd ..'
Then navigate into the data directory:
'cd data'
We need unzip to do the next step:
'apt-get install unzip'
Then download the dataset by:
'bash get_coco_dataset.sh'
This will take a long time, since the dataset is 13GB, So grab a coffee now.


#### Training
Now we can start training the YOLOv3 model on the COCO dataset.
Use the following:
'train.py [-h] [--epochs EPOCHS] [--image_folder IMAGE_FOLDER]
                [--batch_size BATCH_SIZE]
                [--model_config_path MODEL_CONFIG_PATH]
                [--data_config_path DATA_CONFIG_PATH]
                [--weights_path WEIGHTS_PATH] [--class_path CLASS_PATH]
                [--conf_thres CONF_THRES] [--nms_thres NMS_THRES]
                [--n_cpu N_CPU] [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--checkpoint_dir CHECKPOINT_DIR]'


## Mask-RCNN
