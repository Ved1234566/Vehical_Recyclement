import cv2 as cv
import numpy as np
import os

class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}  # Store the center positions of detected objects
        self.id_count = 0  # Keep track of object IDs

    def update(self, detections):
        objects_bbs_ids = []
        new_center_points = {}

        for detection in detections:
            x, y, w, h = detection
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            same_object_detected = False
            for object_id, (px, py) in self.center_points.items():
                distance = np.hypot(cx - px, cy - py)
                if distance < 20:  # Threshold for considering the same object
                    same_object_detected = True
                    new_center_points[object_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, object_id])
                    break

            if not same_object_detected:
                self.id_count += 1
                new_center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])

        self.center_points = new_center_points.copy()
        return objects_bbs_ids


def findObjects(outputs, img, confThreshold, nmsThreshold, tracker):
    hT, wT, _ = img.shape
    bbox, classIds, confs = [], [], []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    detections = []
    if bbox:
        indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append(bbox[i])

    boxes_ids = tracker.update(detections)
    for x, y, w, h, obj_id in boxes_ids:
        cv.putText(img, f'ID {obj_id}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return len(boxes_ids)


def main(video_path=None):
    base_path = r"C:\Users\vedan\Downloads\Junkyard_Cars_Detector-master\Junkyard_Cars_Detector-master\Resources"
    
    if video_path is None:
        video_path = os.path.join(base_path, "Drone.mp4")

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_path}'")
        return

    whT, confThreshold, nmsThreshold = 320, 0.6, 0.2
    tracker = EuclideanDistTracker()

    classesFile = os.path.join(base_path, "coco.names")
    if not os.path.exists(classesFile):
        print(f"Error: Class names file '{classesFile}' not found.")
        return

    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip().split('\n')

    cfg_file = os.path.join(base_path, "custom-yolov4-tiny-detector.cfg")
    weights_file = os.path.join(base_path, "custom-yolov4-tiny-detector_best.weights")

    if not os.path.exists(cfg_file) or not os.path.exists(weights_file):
        print("Error: Model files not found. Please check paths.")
        return

    net = cv.dnn.readNetFromDarknet(cfg_file, weights_file)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    try:
        outputNames = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    except AttributeError:
        outputNames = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or unable to fetch frame.")
            break

        frame = cv.resize(frame, (640, 480))  # Ensuring standard frame size
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outputs = net.forward(outputNames)

        total_objects = findObjects(outputs, frame, confThreshold, nmsThreshold, tracker)
        cv.putText(frame, f'Count: {total_objects}', (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        print(f"Displaying frame... Objects detected: {total_objects}")  # Debugging output
        cv.imshow('Vehicle Counter', frame)

        if cv.waitKey(10) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
