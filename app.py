import cv2
import numpy as np
from scipy.spatial import distance as dist
from flask import Flask, render_template, request
import os

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'videos/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist('file'):
        print(file)
        filename = "video.mp4"
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
    # Initialize video stream
    cap = cv2.VideoCapture('./videos/video.mp4')

    # Initializing threshold values
    confThreshold = 0.3
    nmsThreshold = 0.3
    MinDist = 60

    # Importing coco.names dataset and storing the names in a list
    classesFile = './yolo-coco/coco.names'
    classNames = []
    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    # print(classNames)

    # Path to YOLOv3 model config and weights file
    modelConfiguration = './yolo-coco/yolov4-tiny.cfg'
    modelWeights = './yolo-coco/yolov4-tiny.weights'

    # Setting the model
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    # Setting backend and target
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # Not using CUDA
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def find_objects(img, net, ln, personIdx=0):
        (ht, wt) = img.shape[:2]
        # print(ht,wt)

        # To store person prediction probability, bbox coordinates and centroid of object
        results = []

        # Convert image frames to blob to feed into the model
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)

        # Forward pass
        layeroutputs = net.forward(ln)

        # Initialize lists for detected bboxes, centroid values and
        # probabilities (confidences)
        bbox = []
        centroids = []
        confs = []
        # print(layeroutputs)
        for output in layeroutputs:
            for det in output:
                # To extract class index and associated probability
                scores = det[5:]
                classIdx = np.argmax(scores)
                # print(classIdx)
                conf = scores[classIdx]

                # To ensure detected object is a person and that the
                # probability crosses the confidence threshold value
                if conf > confThreshold:
                    # Getting bbox coordinates
                    box = det[0:4] * np.array([wt, ht, wt, ht])
                    (center_X, center_Y, width, height) = box.astype('int')

                    x = int(center_X - (width / 2))
                    y = int(center_Y - (height / 2))

                    bbox.append([x, y, int(width), int(height)])
                    centroids.append((center_X, center_Y))
                    confs.append(float(conf))

        # Applying non-maximum suppression to remove overlapping bboxes
        indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

        # Atleast one detection
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (bbox[i][0], bbox[i][1])
                (w, h) = (bbox[i][2], bbox[i][3])

                result = (confs[i], (x, y, x + w, y + h), centroids[i])
                results.append(result)

        # Return results that contains person prediction probability,
        # bbox coordinates and centroid of object
        return results

    #fourcc = cv2.VideoWriter_fourcc(*'MG')
    out = cv2.VideoWriter('./static/output.mp4', cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 20, (640,480), isColor=True)

    while True:
        # Get image frames from video stream
        success, img = cap.read()

        # Resizing the img frames
        img = cv2.resize(img, (640,480))

        # Get layer names
        layerNames = net.getLayerNames()
        # print(layerNames)

        # To get only the output layer names from the YOLO network
        outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        # To detect only people
        results = find_objects(img, net, outputNames, personIdx=classNames.index("person"))

        # Warn if the minimum distance between objects is violated
        warning = set()

        # To ensure there are atleast 2 detections
        if len(results) >= 2:
            centroids = np.array([i[2] for i in results])
            Dist = dist.cdist(centroids, centroids, metric="euclidean")

            # Take upper triangular distance matrix and loop
            for i in range(0, Dist.shape[0]):
                for j in range(i + 1, Dist.shape[1]):
                    print(Dist[i, j])
                    if (Dist[i, j] < MinDist):
                        warning.add(i)
                        warning.add(j)

        for (i, (prob, bbox, centroid)) in enumerate(results):
            # Extracting bbox, centroids
            (start_x, start_y, end_x, end_y) = bbox
            (c_X, c_Y) = centroid
            # Setting color to green if detected by default
            color = (0, 255, 0)

            # If distance is violated, send warning by changing color
            # to red
            if i in warning:
                color = (0, 0, 255)

                # Draw bboxes
            cv2.rectangle(img, (start_x - 5, start_y - 5), (end_x + 5, end_y + 5), color, 2)
            # Circle for centroid of the person
            cv2.circle(img, (c_X, c_Y), 3, color, 1)

        out.write(img)
        cv2.imshow('Image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    out.release()

    return render_template("complete.html")


if __name__=="__main__":
    app.run(debug=True)
