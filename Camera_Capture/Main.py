import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import datetime

def write_image_as_csv(image, image_name):
	#get size of the image
	size = image.size

	#open a file to write the pixel data
	with open('.\\output\\csv\\' + image_name + '.csv', 'w+') as f:
		#read each pixel value and write to the file
		x = 0
		y = 0
		for i in range(size):
			f.write('pixel' + str(i + 1))
			if size != i + 1:
				f.write(',')
		f.write('\n')

		for y in range(28):
			for x in range(28):
				p = image[y][x]
				f.write(str(p))
				if (x + 1) * (y + 1) != 28 * 28:
					f.write(',')
		f.write('\n')

def capture_image_and_save_as_csv():
	date = datetime.datetime.now().strftime('_%Y%m%d_%H-%M-%S')
	image_name = 'test' + date
	
	cap = cv2.VideoCapture(0)
	print('Capturing Image')
	if cap.isOpened():
		ret, frame = cap.read()
	else:
		ret = False
		print('ERROR: Videa capture device could not be found.')
		return -1
	
	gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY)
	grayscale = cv2.resize(gray, dsize = (28, 28), interpolation = cv2.INTER_CUBIC)

	write_image_as_csv(grayscale, image_name)
			
	cv2.imwrite('.\\output\\png\\' + image_name + '.png', grayscale)
	cap.release()
	return '.\\output\\' + image_name + '.png'

def main():
	image_path = capture_image_and_save_as_csv()
	if (image_path == -1):
		print("ERROR: Image could not be captured")
	else:
		print("Image saved to: " + image_path)

if __name__ == "__main__":
	main()
