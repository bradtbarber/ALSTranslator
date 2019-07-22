import cv2
import matplotlib.pyplot as plt
import numpy as np
import datetime

print("--------------")

#def rgb2gray(rgb) :
	#return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

def main():
	cap = cv2.VideoCapture(0)

	print("in main")

	if cap.isOpened():
		ret, frame = cap.read()
		print(ret)
		print(frame)
	else:
		ret = False

	date = datetime.datetime.now().strftime("_%Y%m%d_%H-%M-%S")
	
	cv2.imwrite('snapshotCOLOR.png', frame)
	img = cv2.imread('snapshotCOLOR.png')
	gray=cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
	grayscale = cv2.resize(gray, dsize = (28, 28), interpolation = cv2.INTER_CUBIC)
	cv2.imwrite('test' + date + '.png', grayscale)
	cap.release()

if __name__ == "__main__":
	main()
