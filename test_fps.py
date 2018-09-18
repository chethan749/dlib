import numpy as np
import cv2
from fps import FPS
from WebCamVideoStream import WebCamVideoStream as VS


cap = VS(src = 0).start()
fps = FPS()

fps.start()
while True:
	img = cap.read()
	cv2.imshow('Feed', img)
	
	fps.update()
	
	key = cv2.waitKey(1) & 0xFF
	if key == 27 or key == ord('q'):
		break

fps.stop()
cap.stop()
print('With Threading: ', fps.fps())



cap = cv2.VideoCapture(0)
fps = FPS()

fps.start()
while True:
	_, img = cap.read()
	cv2.imshow('Feed', img)
	
	fps.update()
	
	key = cv2.waitKey(1) & 0xFF
	if key == 27 or key == ord('q'):
		break

fps.stop()
cap.release()
print('Without Threading: ', fps.fps())
