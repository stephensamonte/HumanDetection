
from __future__ import print_function
# from imutils.video import WebcamVideoStream
# from imutils.video import FPS

# This is to get FPS function
from Camera import FPS
# This is to get the WebcamVideoStream function 
from Camera import WebcamVideoStream
import argparse
import cv2

# -------------------------------------------------------------------------------------------------

# However, as we\'re about to find out, using the cv2.imshow can substantially decrease our FPS. 
# The cv2.show function is just another form of I/O,
# only this time instead of reading a frame from a video stream, we've instead sending the frame to output on our display. 

# Note: We're also using the cv2.waitKey(1) function here which does add a 1ms delay to our main loop. That said, 
# this function is necessary for keyboard interaction and to display the frame to our screen 
# 
# (especially once we get to the Raspberry Pi threading lessons).

# Code adapted from Tensorflow Object Detection Framework
# https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

# Running the program above to test camera view
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
                help="# of frames to loop over for FPS test")

# Display argument is important for processing video via pipe
ap.add_argument("-d", "--display", type=int, default=-1,
                help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

# Let's move on to the next code block which does no threading and uses 
# 
# blocking I/O when reading frames from the camera stream
# Determines regular image processing without threadding
# grab a pointer to the video stream and initialize the FPS counter
print("[INFO] sampling frames from webcam...")

# Grabs stream
stream = cv2.VideoCapture(0)
# Starts frame counter
fps = FPS().start()

# loop over some frames
while fps._numFrames < args["num_frames"]:
    # grab the frame from the stream and resize it to have a maximum
    # width of 400 pixels
    (grabbed, frame) = stream.read()
    frame = cv2.resize(frame, (1280, 720))
    
    # check to see if the frame should be displayed to our screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()

# Display information
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
stream.release()
cv2.destroyAllWindows()


# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()
 
# loop over some frames...this time using the threaded stream
while fps._numFrames < args["num_frames"]:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
    
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
 
	# # check to see if the frame should be displayed to our screen
	# if args["display"] > 0:
	# 	cv2.imshow("Frame", frame)
	# 	key = cv2.waitKey(1) & 0xFF
 
	# update the FPS counter
	fps.update()
 
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
