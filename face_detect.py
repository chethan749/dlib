import numpy as np
import cv2
import dlib
import argparse
from fps import FPS
from WebCamVideoStream import WebCamVideoStream

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required = True, help = 'face detector path')
ap.add_argument('-i', '--image', help = 'path to input image')
args = vars(ap.parse_args())

FACIAL_LANDMARK_IDXS = {
	"mouth": (48, 68),
	"right_eyebrow": (17, 22),
	"left_eyebrow": (22, 27),
	"right_eye": (36, 42),
	"left_eye": (42, 48),
	"nose": (27, 35),
	"jaw": (0, 17)
}

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	
	return (x, y, w, h)

def shape_to_np(shape):
	coords = np.zeros((68, 2), dtype = np.int32)
	
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	
	return coords

def det_face(image, wait):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	rects = detector(gray, 0)
	shape_np = np.array([])
	
	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape_np = shape_to_np(shape)
		
		x, y, w, h = rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
		
		for (x, y) in shape_np:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	
	if wait:
		cv2.imshow('Output', image)
		
		cv2.waitKey(0)
		cv2.destroyWindow('Output')
	
	return shape_np

def mark_features(image, shape):
	color = [(19, 199, 109), (79, 76, 240), (230, 159, 23), (168, 100, 168), (158, 163, 32), (163, 38, 32), (180, 42, 220)]
	
	overlay = image.copy()
	output = image.copy()
	
	for (i, name) in enumerate(FACIAL_LANDMARK_IDXS.keys()):
		j, k = FACIAL_LANDMARK_IDXS[name]
		pts = shape[j: k]
		
		if name == 'jaw':
			for l in range(1, len(pts)):
				cv2.line(overlay, tuple(pts[l - 1]), tuple(pts[l]), color[i], 2)
		else:
			hull = cv2.convexHull(pts)
			cv2.drawContours(overlay, [hull], -1, color[i], -1)
	
	output = cv2.addWeighted(overlay, 0.7, output, 0.3, 0)
	cv2.imshow('Regions', output)
	
	cv2.waitKey(0)
	cv2.destroyWindow('Regions')

def visualize_regions(image, shape):
	for (i, name) in enumerate(FACIAL_LANDMARK_IDXS.keys()):
		j, k = FACIAL_LANDMARK_IDXS[name]
		pts = shape[j: k]
		
		x, y, w, h = cv2.boundingRect(np.array([pts]))
		roi = image[y: y + h, x: x + w]
		roi = cv2.resize(roi, (int(200 * len(roi[0]) / len(roi)), 200), cv2.INTER_CUBIC)
		
		im_copy = image.copy()
		
		for x, y in pts:
			cv2.circle(im_copy, (x, y), 1, (0, 0, 255), -1)
		
		cv2.imshow('Landmarks', im_copy)
		cv2.imshow('ROI', roi)
		cv2.waitKey(0)
	
	cv2.destroyWindow('Landmarks')
	cv2.destroyWindow('ROI')
		
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

fps = FPS()

vs = WebCamVideoStream(src = 0).start()

if args['image'] is None:
	fps.start()
	while(True):
		image = vs.read()
		image = cv2.resize(image, (int(400 * len(image[0]) / len(image)), 400), cv2.INTER_CUBIC)
		im2 = image.copy()
		shape = det_face(im2, False)
		cv2.imshow('Feed', im2)
		fps.update()
		
		key = cv2.waitKey(1) & 0xFF 
		if key == 27 or key == ord('q'):
			break
		if key == ord('a'):			
			mark_features(image, shape)
			visualize_regions(image.copy(), shape)
	
	fps.stop()
	print(fps.fps())

else:
	image = cv2.imread(args['image'])
	image = cv2.resize(image, (int(500 * len(image[0]) / len(image)), 500), cv2.INTER_CUBIC)
	shape = det_face(image.copy(), True)
	
	mark_features(image, shape)
	visualize_regions(image.copy(), shape)
	
	cv2.waitKey(0)

cv2.destroyAllWindows()
vs.stop()
