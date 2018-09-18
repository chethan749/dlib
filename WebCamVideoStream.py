from threading import Thread
import cv2

class WebCamVideoStream:
	def __init__(self, src):
		self.stopped = False
		self.cap = cv2.VideoCapture(src)
		
		self.grabbed, self.frame = self.cap.read()
		
	def start(self):
		Thread(target = self.update, args = ()).start()
		return self
	
	def update(self):
		while self.grabbed:
			self.grabbed, self.frame = self.cap.read()
			
			if self.stopped:
				return
	
	def read(self):
		return self.frame
	
	def stop(self):
		self.stopped = True
		self.cap.release()
