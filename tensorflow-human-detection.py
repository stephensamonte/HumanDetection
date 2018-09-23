
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time

# Frames per seconds 

#-------------------------------------------------------------------------------------------------

## This is a basic camera capture in open cv 
# # Added to import video from a camera
# cap = cv2.VideoCapture(0)
# # Record video
# # TODO make into while the cap is opened 
# while (True):
#     # read and record frames 
#     ret,img=cap.read()
#     # Show the video and name the video which is wehre frames will be recorded 
#     cv2.imshow('Video Output', img)

#     # Create an escape key to break out of the program 
#     k = cv2.waitKey(10)& 0xff
#     # 27 is the value of the escape key in ascii code 
#     if k==27 : 
#         # exit the loop
#         break

# # remove the cap 
# cap.release()
# # close all windows 
# cv2.destroyAllWindows()


# --------------------------------------------------------------------------------------------------

# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# https://medium.com/@madhawavidanapathirana/real-time-human-detection-in-computer-vision-part-2-c7eda27115c6
class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time: ", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


# ---------------------------------------------------------------------------------------------------

# Program Main
if __name__ == "__main__":
    # Path to model frozen_inference_graph.pb
    model_path = '/Users/samontetan/Documents/GitHub/HumanDetection/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    
    odapi = DetectorAPI(path_to_ckpt=model_path)
    # Threshhold to determine if a human is a human 
    threshold = 0.80
    # Input for a video 
    # cap = cv2.VideoCapture('/Users/samontetan/Documents/GitHub/HumanDetection/test.mp4')

    # Added to import video from a camera
    cap = cv2.VideoCapture(0)

    # this keeps track of the last time a frame was processed
    last_recorded_time = time.time() 

    # Iterate through video frames 
    # TODO add a false condition for loop 
    while True:

        # grab the current time
        curr_time = time.time() 

        # Retrieve a video 
        r, img = cap.read(0)
        # img = cv2.resize(img, (1280, 720))

        # Process the frames 
        boxes, scores, classes, num = odapi.processFrame(img)

        # Tally the number of humans in the current view 
        count = 0

        # it has been at least 2 seconds. Process image every 2 seconds 
        if curr_time - last_recorded_time >= 2.0: 

            # Visualization of the results of a detection.
            for i in range(len(boxes)):
                # Represents a human
                if classes[i] == 1 and scores[i] > threshold:
                    
                    # Location to box object 
                    box = boxes[i]
                    cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)

                    # Add a person to the current view tally 
                    count = count + 1

            # Added Print Statement to display the number of humans in the current view 
            print("Number of Humans: ", count)

            # Save last recorded time as current time 
            last_recorded_time = curr_time

        # Show a frame video 
        cv2.imshow("preview", img)
        # Escape character to close video 
        key = cv2.waitKey(1)
        # Break from loop when the user presses 'q' 
        if key & 0xFF == ord('q'):
            break

