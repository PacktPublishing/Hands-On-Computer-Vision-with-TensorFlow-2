## YOLO V3

This implementation is based on Zihao Zhang's [implementation](https://github.com/zzh8829/yolov3-tf2). Thanks to him!


### Installation

Convert the Darknet model using the following command:

    wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights
    python convert.py --weights weights/yolov3.weights --output weights/yolov3.tf
