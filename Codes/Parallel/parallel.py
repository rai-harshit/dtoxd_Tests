# Standard python libraries : BEGIN
from multiprocessing import Queue
import logging
import cv2
from keras.models import load_model
import numpy as np
# Standard python libraries : END

# Importing Image Classification model : BEGIN
model = load_model("model.h5")
# Importing Image Classification model : END

# Importing Frame Extraction Libary : BEGIN
# This is a custom library specially built for extracting frames from video using ffmpeg
# and passing them to a shared queue using shared pipe instead of writing those frames 
# on the hard-disk.
# For more info about the code, follow the link : https://github.com/chexov/image2pipe
import image2pipe
# Importing Frame Extraction Libary : END

# Setting basic logging configurations : BEGIN
logging.basicConfig()
# Setting basic logging configurations : BEGIN

# Change the values in the FOR loop and "images_from_url()" method accordingly.
# Name of the file : Duration in seconds 
# got.mkv : 3875
# sm.mkv 7010
# twws.mp4 : 10792

# This code creates a thread and begins frame extraction : BEGIN
q = Queue()
decoder= image2pipe.images_from_url(q,"twws.mp4",fps="1",scale=(300,300))
decoder.start()
# This code creates a thread and begins frame extraction : END

# This code takes frames from the shared queue, passes it to the image classification model,
# saves the prediction in a list : BEGIN
l = [] 
for i in range(10792):
	fn, img = q.get()
	image = np.reshape(img,(1,300,300,3))
	p = model.predict(image)[0]
	l.append(p)