# WARNING THIS REPO IS ABOUT 1GB. CLONE AT YOUR OWN RISK!

# Phase 1

In phase 1, you will be using the LISA dataset in order to make a vehicle localizer using Pytorch. Specifically, you will take the Resnet-18 pretrained model, and modify it so that the final output layer is a bounding box regressor.

## Phase 1 Questions

_Q1\. How does your choice of loss affect the accuracy of your bounding box regressor. Specifically, what is the accuracy difference between L1 or L2 loss? Which works better and why? (To properly answer this question you will need to understand what is happening both conceptually as well have done some experimentation on your model)._

_Q2\. What is the purpose of IoU? Why is it so important in object detection?_

_Q3\. The current model in this phase can only do single object detection. How would you transform this model to handle multiple object detection? (There are multiple valid answers to this. You should compare the merits of each method.)_

# Phase 2

In phase 2, you will be comparing how YOLO-based unified regression models do against Region Proposal Based networks on the COCO dataset. Specifically, you will need to train models using both YOLOv3 and Mask-RCNN.

## Motivation

The goal of this phase is to familiarize how to benchmark the performance of state of the art deep learning algorithms on common datasets. In industry it is very common to productionize the best open source models rather than spending time developing a custom architecture as the development times are very expensive and its possible to make a very accurate model if you have enough data.

## Assignment Details

### Model Creation

In this assignment you will need to produce two object detection models. For benchmark purposes you will need to have two models trained on the COCO dataset. The algorithms you will use for the object detection model are YOLOv3 and Mask-RCNN. Both algorithms have multiple open source implementations. It is up to you to choose the open source implementation for both algorithms you would like to use. There are docker images on both Kronos and Atlas that will allow you to run all of the implementations linked in the README. You will need to figure out how to train the COCO dataset. However, you do not need to go through the entire training process for both these models. Since we are so restricted in terms of compute resources, it would be impossible to finish this lab in a week because for each student it could take anywhere from 2-3 days to train both models on the entire COCO dataset. Instead, you can use pre-trained weights on the COCO dataset, as you will need to use one of the two models to produce a video for phase 3\. One thing to note, is that the COCO dataset has more than just vehicles, but other various objects. However, your detector should suppress drawing bounding boxes for any non-vehicle objects.

### Code Documentation

Your report should include detailed instructions on how to run and train both models. This includes the proper directory structure, command line inputs, python scripts, docker setup, etc. This is to help you get into the habit of creating good documentation to replicate your results. Your documentation should also show how to use your model weights to find all vehicles (not just cars, but trucks, motorcycles, etc.) in a random image.

### Report

Your report should argue for the use of one of these models to be used as a vehicle detector. Your report will be graded on the quality of your argument for one model vs the other. Your report should discuss the accuracy of the model, the speed at which each model runs on a single GPU, and any other metrics which you consider important to a vehicle detection model. Your answer should be justified with statistics from your experimentation with each algorithm as well as from your understandings of how each model works. This can be done by citing knowledge from the original papers themselves, statistics, as well as knowledge, you have gained from class. Your report should provide a very clear argument as to why one model is better than the other, and you must be very explicit as to the model you believe works better for vehicle detection.

## Links

### Mask-RCNN Implementations

[Pytorch](https://github.com/multimodallearning/pytorch-mask-rcnn)

Docker Image: pytorch/pytorch

[Tensorflow](https://github.com/matterport/Mask_RCNN)

Docker Image: tensorflow/tensorflow

### YOLOv3 Implementation

YOLOv3 Code: [Pytorch](https://github.com/ultralytics/yolov3)

Docker Image: pytorch/pytorch

[Darknet Original Implementation](https://pjreddie.com/darknet/yolo/)

Docker Image: loretoparisi/darknet

[COCO Dataset](http://cocodataset.org/#home)

# Phase 3

After determining which model works better for vehicle detection,you must use that model to detect all vehicles in the Udacity highway video from the lane line detection lab. Please include the final video with both the detected vehicles and the detected lane lines. Please keep in mind that your final video should only include vehicles on the road. If the video includes detections of other objects it will be considered incorrect.

Bonus: Output the real world coordinate locations for all the vehicles detected (you just need to provide the X,Y coordinates).

# Final Submission Details

Your final submission must include the following things: 1\. Your filled out Python notebook for Phase 1.

1.  Full instructions in a markdown file for training/inference of both Mask-rcnn and YOLOv3 as specified in the Phase 2 instructions.

2.  A report with the answers for the Phase 1 questions, and your report for Phase 2.

3.  A video with all detected objects as well as lane line detections in the Udacity lane line detection video.