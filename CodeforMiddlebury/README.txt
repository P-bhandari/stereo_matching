######Stereo Matching Code for Middlebury Dataset#######

How to set up - 

1) Install the following libraries :- 
	i) opencv3.0 or greater
	ii) install opencv along with the opencv_contrib libraries
	iii)install boost libraries


2) Copy left and right Images from middlebury dataset(/dataset/2014/MiddEval3 Data (2)/trainingQ/) to the bin folder inside StereoCode
3) Change the name of the files accordingly for each dataset i.e(2001,2003,2014). 
4) Copy the ground truth image(/dataset/2014/MiddEval3/trainingQ/) to the bin folder inside StereoCode.
5) Inside the ground truth folder copy the value inside "MaxDisp.txt" and paste the value to the numerator of the maxDisp variable in example.cpp . ##(The value is only used for comparison, the  maximum disparity for the code is set to the upper limit (i.e 200 or ..) depending on the dataset)##
 
6) Adjust the  parameters according to the output required.



Steps to run the code :- 

1) Open a terminal 
2) cd ../CodeforKITTI/
3) mkdir build
4) cd build/
5) cmake ..
6) make
7) cd ..
5) cd bin/
6) ./example
7) plot.m contains the output for all census window types.
