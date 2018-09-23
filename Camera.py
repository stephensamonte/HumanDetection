from threading import Thread
import cv2

import datetime

#-------------------------------------------------------------------------------------------------

# Code adapted from Tensorflow Object Detection Framework
# https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

# Determine frames per seconds
class FPS:
    
    # Constructor for FPS
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        
        # Start of frame read
        self._start = None
        # End of frame read
        self._end = None
        
        # Number of frames that were read between start and end
        self._numFrames = 0
    
    # Starts the timer
    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self
    
    # Saves the end time
    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()
    
    # Increment the frame count
    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1
    
    # Determine the amount of time between start and stop time
    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()
    
    # Determine frames per seconds
    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


#-------------------------------------------------------------------------------------------------

# Code adapted from Tensorflow Object Detection Framework
# https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

# Camera View stream
class WebcamVideoStream:
    
    # Constructor
    # src = 0 is the webcamera
    # If src is a string then it is assumed to be a video file path
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        # Returns a pointer to video stream
        self.stream = cv2.VideoCapture(src)
        # Read the stream. Get the next available frame
        (self.grabbed, self.frame) = self.stream.read()
        
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    # Start the stream
    def start(self):
        
        # start the thread to read frames from the video stream
        # Update method is placed in a separate thread
        Thread(target=self.update, args=()).start()
        return self
        
    def update(self):
        # keep looping infinitely until the thread is stopped
        # Continually reads the next available frame
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame
        
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
