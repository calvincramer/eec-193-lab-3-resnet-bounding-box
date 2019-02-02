# Train YOLOv3 and Mask-RCNN with COCO database

### Install docker
Install docker on your computer. Depending on your operating system, follow the given guides:
* Linux ->
* Mac ->
* Windows ->

After that, if you would like to train on your Nvidia GPU, use the official [guide](https://github.com/NVIDIA/nvidia-docker#quick-start)
to install `nvidia-docker`

### Get Ubuntu 16.0 docker image
Get the docker image for the latest version of ubuntu using:

`$ docker pull ubuntu:16.04`

Then make sure the image was pulled correctly by using:

`$ docker images`

And you should see the `ubuntu` image with the tab `16.04` in the list.


### Make container, start and attach
Make the ubuntu docker container using:

`$ docker container create -it --name <container_name> ubuntu /bin/bash`

Where `<container_name>` can be whatever you want the name of the container to be. Then if you type:

`$ docker ps -a`

You should see `<container_name>` in the list of all containers
Then start and attach to the container:

```
$ docker start <container_name>
$ docker attach <container_name>
```

If you would like to train on the GPU, instead run the container doing the following (not necessary to do until you are actually going to train):

```
$ nvidia-docker start <container_name>
$ nvidia-docker attach <container_name>
```

You should now be inside the running docker container.
If you don't see a shell prompt, press ctrl+c.

### Install git
First of all, update the libraries that came within the container:

`$ apt-get update`

Now install git:

`$ apt-get install git`

### Install curl
We need this to install anaconda

`$ apt-get install curl`

### Install anaconda
We will be following this [guide](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04). You can stop after the `conda list` step.

The guide is summarized below:

Run these commands line by line:
```
$ cd /tmp
$ curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
$ bash Anaconda3-5.0.1-Linux-x86_64.sh
```

Then press enter until you get to then end of the liscence, they type `yes`



## YOLOv3
We will be using eriklindernoren's implementation of YOLOv3:

Clone the git repository

`$ git clone https://github.com/eriklindernoren/PyTorch-YOLOv3`

Were going to use pip to install numpy, torch, torchvision, pillow, and matplotlib:

`$ apt-get install python3-pip`

Navigate into the repository we just cloned:

`$ cd PyTorch-YOLOv3`

Install the dependencies needed for PyTorch-YOLOv3:

`$ pip3 install -r requirements.txt`

From some reason if torch is not installed if you type `conda list` then install torch using:

`$ conda install -c soumith pytorch`

And torch vision using:

`$ conda install -c pytorch torchvision`

If you want to use the YOLO model with pretrained weights, follow these steps:
```
$ apt-get install wget
$ cd weights
$ bash download_weights.sh
```

#### Now dowload the COCO dataset:
Make sure you are in the PyTorch-YOLOv3 base directory. If you are already in the weights folder just do

`$ cd ..`

Then navigate into the data directory:

`$ cd data`

We need unzip to do the next step:

`$ apt-get install unzip`

Then download the dataset by:

`$ bash get_coco_dataset.sh`

This will take a long time, since the dataset is 13GB, so grab a coffee now.

Now before we train, we need to solve some error with QT:

`$ apt install libgl1-mesa-glx`


#### Training
Now we can start training the YOLOv3 model on the COCO dataset.
Make sure you are in the `PyTorch-YOLOv3` directory then do the following:

`$ python train.py`

YOLOv3 should use the COCO dataset and start traing. You should see the following if it is training successfully:

```
[Epoch 0/30, Batch 0/7329] [Losses: x 0.287326, y 0.253497, w 3.449932, h 5.507586, conf 4.123625, cls 0.821795, total 14.443762, recall: 0.00402, precision: 0.00001]
[Epoch 0/30, Batch 1/7329] [Losses: x 0.275898, y 0.270465, w 5.844891, h 11.642635, conf 4.051937, cls 0.819050, total 22.904877, recall: 0.00000, precision: 0.00000]
[Epoch 0/30, Batch 2/7329] [Losses: x 0.290545, y 0.242395, w 10.648568, h 11.280510, conf 3.956580, cls 0.816539, total 27.235136, recall: 0.01010, precision: 0.00029]
[Epoch 0/30, Batch 3/7329] [Losses: x 0.276022, y 0.271325, w 5.613185, h 5.836898, conf 3.962628, cls 0.816571, total 16.776630, recall: 0.00000, precision: 0.00000]
....
```

Now you just need to wait until it is finished training, which will be a while.

### Run a test image, and find all vehicles





## Mask-RCNN

We will be an implementation from Facebook: [repo](https://github.com/facebookresearch/maskrcnn-benchmark)

First download the docker file from the repository. You can either make an enviornment with or without jupyter notebook:
* With juypter: maskrcnn-benchmark/docker/docker-jupyter/Dockerfile
* Without jupyer: maskrcnn-benchmark/docker/Dockerfile

Save the dockerfile as `Dockerfile`

WARNING: make sure to match the cuda version and cudann version in the first few lines of the dockerfile with the machine you are running. Check your machine cuda version using `nvcc --version`

If you are using the jupyter notebook dockerfile, make sure you also download the `jupyter_notebook_config.py` file.

Next we will build the docker image:

`$ docker build -f DockerfileMaskRCNN -t mask-rcnn:facebook-mask-rcnn .
`

If the dockerfile was build sucessfully, typing `docker images`, you should see an image with name `mask-rcnn` and tag `facebook-mask-rcnn`

Next build the docker container:
If you are using port forwarding to access the jupyter notebook then use the `-p 8888:8888` flag, otherwise, don't use it:

`$ nvidia-docker run --shm-size 8G -it --name <container_name> -p 8888:8888 --entrypoint=/bin/bash mask-rcnn:facebook-mask-rcnn`


Then exit the container: `exit`


### Download the COCO dataset

We will use a shell script to make this easier:

Download this file: 

`$ https://github.com/pjreddie/darknet/tree/master/scripts/get_coco_dataset.sh`

Then copy the file into the docker container:

`$ nvidia-docker cp get_coco_dataset.sh <container_name>:/notebooks/get_coco_dataset.sh`

Then start and attach to the container:

```
$ nvidia-docker start <container_name>
$ nvidia-docker attach <container_name>
```

Then run the get COCO script:

```
$ apt-get install wget
$ apt-get install unzip
$ bash get_coco_dataset.sh
```

Then we will link the COCO dataset with the model.
Make sure you are in the `/notebooks/maskrcnn-benchmark/` folder

```
# symlink the coco dataset
mkdir -p datasets/coco
ln -s /notebooks/coco/annotations datasets/coco/annotations
ln -s /notebooks/coco/images/train2014 datasets/coco/train2014
ln -s /notebooks/coco/images/test2014 datasets/coco/test2014  -- DONT HAVE
ln -s /notebooks/coco/images/val2014 datasets/coco/val2014
```

If you mess up and one of the `ln` commands says the file already exists, use the `-f` flag to replace the destination link.

We need one more file for training that was not downloaded. Download it from here:

`https://dl.fbaipublicfiles.com/detectron/coco/coco_annotations_minival.tgz`

This will download a .tgz file. Extract this and get the `instances_valminusminival2014.json` file.
Do this with:

`tar -xzf coco_annotations_minival.tgz`

This creates a `annotations` folder. Inside this folder is the `instances_valminusminival2014.json` file.

Place this file in `/notebooks/maskrcnn-benchmark/datasets/coco/annotations`


#### Training

Make sure you are in the `/notebooks/maskrcnn-benchmark` folder, then download from:

`python tools/train_net.py --config-file configs/e2e_faster_rcnn_R_50_C4_1x.yaml`




### Run a test image, and find all vehicles








YOLO
detect.py
python3 detect.py -image_folder /data/samples
