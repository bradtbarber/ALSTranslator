import cv2
import matplotlib.pyplot as plt
import numpy as np

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

	
	cv2.imwrite('snapshotCOLOR.png', frame)
	img = cv2.imread('snapshotCOLOR.png')
	#colorpic = cv2.cvtColor( img, cv2.COLOR_BGR2RGB)
	#cv2.imwrite('snapshotCOLOR.png', colorpic)
	gray=cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
	grayscale = cv2.resize(gray, dsize = (28, 28), interpolation = cv2.INTER_CUBIC)
	cv2.imwrite('snapshotGRAY.png', grayscale)
	#img1 = cv2.imread(frame)
	#cv2.imwrite('hand.png', img1)
	

	#plt.imshow(colorpic)
	#plt.title('Color Image RGB')
	#plt.xticks([])
	#plt.yticks([])
	#plt.show()

	cap.release()


	#cv2.imwrite('hand-scaled.png',img1)


	#gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	#lum_img = rgb2gray(img1)
	#plt.imshow(gray)
	#plt.title('Color Image Gray')
	#plt.show()

if __name__ == "__main__":
	main()