OVST.ods : File containing the benchmarking of offline video speed test.

Use the file 'ovst.py' to test the speed of frame extraction and content prediction.
Before running the file, make sure that the following libraries are installed:
	a: image2pipe
	b: Keras + Tensorflow
	c: OpenCV
	d: Numpy

Follow the link below to download the 3 video files on which benchmarking is performed:
	a: got.mkv : 274 MB : Link https://drive.google.com/file/d/1xq-s3XvSQuYokJO1VkENFtHsgFeaRjnA/view?usp=sharing
	b: sm.mkv : 1 GB : Link https://drive.google.com/file/d/1I-OpvfezXYxocUVNk9klfkIQw9u6yi-0/view?usp=sharing
	c: twws.mp4 : 2.2 GB : Link https://drive.google.com/file/d/1qohO055YR_9po41qQm2UoBn165-9stid/view?usp=sharing

After downloading the videos, make appropriate changes in the ovst.py file to make sure the filename and the duration parameters are set.
After making all the changes, run the following command:
	time python ovst.py

After the processing completes, you will see 3 parameters:
	a: real
	b: user
	c: sys
Fill in the above values in the OVST.ods file corresponding to your machine and update the benchmark.