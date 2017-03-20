from keras.models import load_model
from keras.preprocessing.image import load_img, array_to_img, img_to_array
import time
import numpy as np
import cv2
import argparse

start = time.time()

print("Input model : ")
modelLink = input()
model = load_model(modelLink)
point1 = time.time()
print("load model")
print("load complete")
print(point1 - start)
while (True):
	print("Insert image:")
	imgLink = input()
	try:
		img = load_img(imgLink, target_size=(299,299))
	except:
		continue
	img = img_to_array(img)
	imgList = np.expand_dims(img, axis=0)
	prediction = model.predict(imgList, batch_size=1)
	if prediction[0][0] > prediction[0][1]:
		label = 'dog'
	else:
		label = 'cat'
	orig = cv2.imread(imgLink)
	cv2.putText(orig, label, (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
	cv2.imshow('result', orig)
	print(prediction)
	print(imgList.shape)
	point2 = time.time()
	print(point2 - point1)	
	cv2.waitKey(0)
