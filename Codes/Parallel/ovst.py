from multiprocessing import Queue
import logging
import cv2
from keras.models import load_model
import numpy as np
model = load_model("model.h5")

import image2pipe

logging.basicConfig()

# Change the values in the FOR loop and images_from_url() accordingly.
# Name of the file : Duration in seconds 
# got.mkv : 3875
# sm.mkv 7010
# twws.mp4 : 10792

q = Queue()
decoder= image2pipe.images_from_url(q,"twws.mp4",fps="1",scale=(300,300))
decoder.start()

for i in range(10792):
	fn, img = q.get()
	image = np.reshape(img,(1,300,300,3))
	p = model.predict(image)