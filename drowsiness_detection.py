from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(e):

	a = dist.euclidean(e[1], e[5])
	b = dist.euclidean(e[2], e[4])

	c = dist.euclidean(e[0], e[3])
	ear = (a+b)/(2.0*c)

	return ear

#argument parser
ag = argparse.ArgumentParser()
ag.add_argument("-p", "--shape-predictor", required = True, help = "path to facial landmark predictor")
ag.add_argument("-a", "--alarm", type = str, default = "", help = "path alarm .wav file")
ag.add_argument("-w", "--webcam", type = int, default = 0, help = "index of webcam on system")

args = vars(ag.parse_args())

#defining the threshold
eye_ar_thresh = 0.3
eye_ar_frames = 40
blink = 0

counter = 0
alarm_on = False

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src = args["webcam"]).start()
time.sleep(1.0)

def sound_alarm(path):
	playsound.playsound(path)

# loop over the frames from the video stream
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width = 450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 0)

	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		lefteye = shape[lStart:lEnd]
		righteye = shape[rStart:rEnd]
		leftear = eye_aspect_ratio(lefteye)
		rightear = eye_aspect_ratio(righteye)

		ear = (leftear+rightear)/2.0

		leftEyeHull = cv2.convexHull(lefteye)
		rightEyeHull = cv2.convexHull(righteye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear<eye_ar_thresh:
			counter += 1
			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if counter >= eye_ar_frames:
				blink +=1
				# if the alarm is not on, turn it on
				if not alarm_on:
					alarm_on = True

					if args["alarm"] != "":
						t = Thread(target = sound_alarm,
									args = (args["alarm"],))
						t.deamon = True
						t.start()


				cv2.putText(frame, "Drowsiness Detected!", (10, 30), 
							cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
				

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			counter = 0
			alarm_on = False

			cv2.putText(frame, "Blinks: {}".format(blink), (200,20), 
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

		cv2.putText(frame, "EAR {:.2f}".format(ear), (300,30), 
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
			
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord('q'):
		break

cv2.destroyAllWindows()
vs.stop()








