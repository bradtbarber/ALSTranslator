import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import datetime

def capture_image():
	date = datetime.datetime.now().strftime('_%Y%m%d_%H-%M-%S')
	image_name = 'test' + date
	
	cap = cv2.VideoCapture(0)
	print('Capturing Image')
	if cap.isOpened():
		ret, frame = cap.read()
		print(ret)
		print(frame)
	else:
		ret = False
		print('ERROR: Videa capture device could not be found.')
		return -1
	
	#cv2.imwrite(image_name + 'COLOR.png', frame)
	#img = cv2.imread(image_name + 'COLOR.png')
	gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY)
	grayscale = cv2.resize(gray, dsize = (28, 28), interpolation = cv2.INTER_CUBIC)
	print("Grayscale version of image:")
	print(grayscale)

	#get a tuple of the x and y dimensions of the image
	size = grayscale.size
	print("size: " + str(size))

	#open a file to write the pixel data
	with open('.\\output\\' + image_name + '.csv', 'w+') as f:
		#read the details of each pixel and write them to the file
		x = 0
		y = 0
		for i in range(size):	
			print ("x,y: " + str(x) + ',' + str(y))
			p = grayscale[x][y]
			f.write('{0}'.format(p))

			if x + 1 == 28:
				x = 0
				y = y + 1
				f.write('\n')
			else:
				x = x + 1
				f.write(',')
			
	cv2.imwrite('.\\output\\' + image_name + '.png', grayscale)
	cap.release()
	return '.\\output\\' + image_name + '.png'

def main():
	image_path = capture_image()
	if (image_path == -1):
		print("ERROR: Image could not be captured")
	else:
		print("Image saved to: " + image_path)

if __name__ == "__main__":
	main()
