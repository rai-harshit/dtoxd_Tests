from keras.models import load_model
import cv2
from tqdm import tqdm
import numpy as np
import time
import shutil
import os


rootd = "/home/g_host/Desktop/dtoxd/hr/ecc/"
model_file = rootd+"model.h5"
frames_store = rootd+"storage/frames/"
video = rootd+"test.mp4"
sensitivity = 0.05

os.system('ffmpeg -i {} -vf  "select=gt(scene\,{}), scale=300:300, showinfo" -vsync vfr {}%01d.jpg 2>&1 | grep -o "pts_time:[0-9.]*" > timestamp.log'.format(video,sensitivity,frames_store))

# print("Frame Extraction Completed Successfully")
frames = os.listdir(frames_store)
# frames.sort()
# print(frames)

list1 = []
for fms in frames:
	fms = int(fms.split(".")[0])
	list1.append(fms)
list1.sort(key=int)

# print(list1)

raw_ts = []
with open('timestamp.log') as fp:
    for line in fp:
    	raw_ts.append(line.rstrip("\n"))
timestamps = []
for ts in raw_ts:
	ts = ts.split(":")[1]
	timestamps.append(ts)
test_data = []
for frame in tqdm(list1):
	# print(frames_store+str(frame)+".jpg")
	im = cv2.imread(frames_store+str(frame)+".jpg")
	# print(im.shape)
	test_data.append(im)
test_data = np.array(test_data)
model = load_model(model_file)
result = []

predict_start = time.time()
predictions = model.predict(test_data)
predict_end = time.time()

# print(predictions)
for prediction in predictions:
	if prediction[0]>prediction[1]:
		result.append(1)
	else:
		result.append(0)
explicit_durations = []
for data in zip(list1,timestamps,result):
	if data[2] == 1:
		explicit_durations.append(data[1])
		# print(data[0],data[1])
if len(explicit_durations) > 0:
	conditions = []
	last_explicit_duration = 0
	for duration in explicit_durations:
		if float(duration)-last_explicit_duration<1:
			conditions.append("between(t,{},{})".format(float(last_explicit_duration),float(duration)+0.1))
		else:
			conditions.append("between(t,{},{})".format(float(duration),float(duration)+0.1))
		last_explicit_duration = float(duration)
	final_condition = "+".join(conditions)
	# print(final_condition)
	os.system("ffmpeg -i {} -q:v 1 -qmin 1 -filter_complex \"boxblur=90.0:enable='if({},1,0)\" -codec:ac copy test.avi | vlc test.avi".format(video,final_condition))
	print(final_condition)
# time2 = time.time()
# total = time2-time1
#
# print(total)
ext_time = ext_end - ext_start
prediction_time = predict_end - predict_start

print(ext_time)
print(prediction_time)
