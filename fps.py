from datetime import datetime as dt

class FPS:
	def __init__(self):
		self._start = None
		self._stop = None
		self._frames = 0
	
	def start(self):
		self._start = dt.now()
	
	def stop(self):
		self._stop = dt.now()
	
	def update(self):
		self._frames += 1
	
	def elapsed(self):
		return (self._stop - self._start).total_seconds()
	
	def fps(self):
		return self._frames / self.elapsed()
