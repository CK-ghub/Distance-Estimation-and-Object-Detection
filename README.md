# Distance-Estimation-and-Object-Detection
Distance Estimation and Object Detection using YOLOv4 DarkNet, OpenCV and Flask.

## Project Requirements:
	1. NumPy
	2. CV2 
	3. SciPy 
	4. flask
	5. Pre-trained YOLOv4-tiny weights, configuration file 
	6. coco.names dataset the YOLO object detection model is trained on
	7. CCTV Footages
    
## Model Used:
    YOLOv4 DarkNet Object Detector 

## Algorithms:
	1. Intersection Over Union (IOU) - The calculation is done to measure the overlap between two propasals. The NMS uses this as a metric to predict bounding boxes to compute the ratio between the area of intersection and the area of union to predict the percentage or the measure of overlap of the bounding boxes, generally when the model detects the same object more than once. 
	
	2. Non-Maximum Suppression (NMS) - Takes a threshold value as a parameter to eliminate those boxes which have IOU greater than the threshold until no overlapping boxes are left starting with proposals with highest confidence or probability score.

## Resources:
	1. CCTV footages
	2. Pretrained YOLOv4-tiny weights - 	https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
	3. YOLOv4-tiny Config file - https://github.com/kiyoshiiriemon/yolov4_darknet
	4. YOLO openCV documentation - https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
