import csv
import msvcrt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import datetime

def write_csv_columns(image_name):
	#open a file to write the column names
	with open('.\\output\\csv\\' + image_name + '.csv', 'w+') as f:
		#write column names
		for i in range(784):
			f.write('pixel' + str(i + 1))
			if 784 != i + 1:
				f.write(',')
		f.write('\n')

def write_image_as_csv(image, image_name):
	#open a file to write the pixel data
	with open('.\\output\\csv\\' + image_name + '.csv', 'a') as f:
		#read each pixel value and write to the file
		x = 0
		y = 0
		for y in range(28):
			for x in range(28):
				p = image[y][x]
				f.write(str(p))
				if (x + 1) * (y + 1) != 28 * 28:
					f.write(',')
		f.write('\n')

def capture_image_and_save_as_csv():
	#generate file name for csv output
	date = datetime.datetime.now().strftime('_%Y%m%d_%H-%M-%S')
	image_name = 'test' + date

	#retreive camera
	cap = cv2.VideoCapture(0)

	#prepare output csv columns
	write_csv_columns(image_name)

	picture_num = 0

	#inform user that program is ready to take a picture
	print('Ready to begin.\n' + 
		'Please press \'spacebar\' to take pictures.\n' + 
		'Press \'esc\' when you are done.')

	while 1:
		check, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.rectangle(gray, (80,80), (550, 400), (255,255,00),2)
		cv2.imshow("Capturing", gray)
		
		key = cv2.waitKey(1)
		
		#wait for keyboard interrupt
		if key == 32:
			print('Capturing Image...\n')
			if cap.isOpened():
				ret, frame = cap.read()
			else:
				ret = False
				print('ERROR: Video capture device could not be found.')
				return -1
		
			#convert image to greyscale and scale to 784 pix
			grayscale = cv2.resize(
				cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY), 
				dsize = (28, 28), 
				interpolation = cv2.INTER_CUBIC
				)

			#write picture pix values as new row to csv
			write_image_as_csv(grayscale, image_name)

			#write image to output directory for debugging and verification		
			cv2.imwrite('.\\output\\png\\' + image_name + '_' + str(picture_num) + '.png', grayscale)
			picture_num = picture_num + 1

			#inform user that program is ready for the next input
			print('OK: Ready to take next picture.')
		
		elif key == 27:
			print('Exiting...\n')
			break
	
	#release camera and return path to csv
	cap.release()
	return '.\\output\\csv\\' + image_name + '.csv'

def main():
	image_path = capture_image_and_save_as_csv()
	if (image_path == -1):
		print("ERROR: Image could not be captured")
	else:
		print("Images saved to: " + image_path)

if __name__ == "__main__":
	main()
