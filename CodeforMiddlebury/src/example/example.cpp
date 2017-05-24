	/*
 * Author: Konstantin Schauwecker
 * Year:   2012
 */
 
// This is a minimalistic example on how to use the extended
// FAST feature detector and the sparse stereo matcher.

#include <opencv2/opencv.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <vector>
#include <iostream>
#include <sparsestereo/exception.h>
#include <sparsestereo/extendedfast.h>
#include <sparsestereo/stereorectification.h>
#include <sparsestereo/sparsestereo-inl.h>
#include <sparsestereo/census-inl.h>
#include <sparsestereo/imageconversion.h>
#include <sparsestereo/censuswindow.h>
#include <fstream>
#include <sparsestereo/sparseToDense.h>
#include <sparsestereo/keypoints.h>
using namespace std;
using namespace cv;
using namespace sparsestereo;
using namespace boost;
using namespace boost::posix_time;

string getFileName(int lexer);
void comparingOfDisparityMap(Mat disp,vector<SparseMatch> correspondences, int lexer,string LeftpathName, string RightpathName, string fileType,string filename,  int censusWinType, int combination);
void occGTCompare(Mat disp,vector<SparseMatch> correspondences, int lexer, string fileType,string filename,  int censusWinType, int combination);
void nonoccGTCompare(Mat disp, vector<SparseMatch> correspondences, int lexer, string fileType, string filename, int censusWinType, int combination);
Mat disparityMapOfSGBM(cv::Mat_<unsigned char> leftImg,cv::Mat_<unsigned char>  rightImg);
Mat postProcessingAndComparing(Mat disp, Mat leftImage);
Mat filterSpeckles1( InputOutputArray _img, double _newval, int maxSpeckleSize,double _maxDiff, InputOutputArray __buf);
void filterSpeckles1Impl(cv::Mat& img, int newVal, int maxSpeckleSize, int maxDiff, cv::Mat& _buf);
void printToAFile(int lexer);
void nonoccGTCompareSparse( vector<SparseMatch> correspondences, int lexer, string fileType, string filename, int censusWinType, int combination);
void occGTCompareSparse( vector<SparseMatch> correspondences, int lexer, string fileType,string filename,int censusWinType,int combination);
void nonoccSgbmGTCompareSparse(vector<SparseMatch> correspondences, int lexer, string fileType,string filename, Mat SGBMDisp,int censusWinType,int combination);
void occSgbmGTCompareSparse(vector<SparseMatch> correspondences, int lexer, string fileType,string filename, Mat SGBMDisp,int censusWinType,int combination);
void SgbmGTCompare(vector<SparseMatch> correspondences, int lexer, string fileType,string filename, Mat SGBMDisp ,int censusWinType,int combination);
void occSgbmGTCompare(vector<SparseMatch> correspondences, int lexer, string fileType,string filename, Mat SGBMDisp,int censusWinType,int combination);
void sparseToDenseConversion(Mat leftColoredImg, Mat rightColoredImg, Mat& disparityImage1,Mat labels,Mat leftMask, vector<SparseMatch> correspondences, int edgeNo);
void nonoccSgbmGTCompare(vector<SparseMatch> correspondences, int lexer, string fileType,string filename, Mat SGBMDisp,int censusWinType,int combination);


const double PI = 3.14159;
const int numOfCensusTypes = 10;
const int maxCombinations = 1;
int sparse = 0;
float OCCcannyResultwithinError1[maxCombinations][numOfCensusTypes][500];
float OCCcannyResultwithinError2[maxCombinations][numOfCensusTypes][500];
float OCCcannyResultwithinError3[maxCombinations][numOfCensusTypes][500];
float OCCcannyResulterrorMoreThan3[maxCombinations][numOfCensusTypes][500];
int OCCcannyResultNumberOfFeatures[maxCombinations][numOfCensusTypes][500];
float OCCcannyResultinvalidPoint[maxCombinations][numOfCensusTypes][500];
float OCCcannyResultwithinErrorHalf[maxCombinations][numOfCensusTypes][500];
// float OCCcannyResulterrorMoreThanHalf[maxCombinations][numOfCensusTypes][500];

float NOCcannyResultwithinError1[maxCombinations][numOfCensusTypes][500];
float NOCcannyResultwithinError2[maxCombinations][numOfCensusTypes][500];
float NOCcannyResultwithinError3[maxCombinations][numOfCensusTypes][500];
float NOCcannyResulterrorMoreThan3[maxCombinations][numOfCensusTypes][500];
int   NOCcannyResultNumberOfFeatures[maxCombinations][numOfCensusTypes][500];
float NOCcannyResultinvalidPoint[maxCombinations][numOfCensusTypes][500];
float NOCcannyResultwithinErrorHalf[maxCombinations][numOfCensusTypes][500];
//float NOCcannyResulterrorMoreThanHalf[maxCombinations][numOfCensusTypes][500];


float OCCSGBMResultwithinError1[maxCombinations][numOfCensusTypes][500];
float OCCSGBMResultwithinError2[maxCombinations][numOfCensusTypes][500];
float OCCSGBMResultwithinError3[maxCombinations][numOfCensusTypes][500];
float OCCSGBMResulterrorMoreThan3[maxCombinations][numOfCensusTypes][500];
int   OCCSGBMResultNumberOfFeatures[maxCombinations][numOfCensusTypes][500];
float OCCSGBMResultinvalidPoint[maxCombinations][numOfCensusTypes][500];
float OCCSGBMResultwithinErrorHalf[maxCombinations][numOfCensusTypes][500];
//float OCCSGBMResulterrorMoreThanHalf[maxCombinations][numOfCensusTypes][500];

float NOCSGBMResultwithinError1[maxCombinations][numOfCensusTypes][500];
float NOCSGBMResultwithinError2[maxCombinations][numOfCensusTypes][500];
float NOCSGBMResultwithinError3[maxCombinations][numOfCensusTypes][500];
float NOCSGBMResulterrorMoreThan3[maxCombinations][numOfCensusTypes][500];
int   NOCSGBMResultNumberOfFeatures[maxCombinations][numOfCensusTypes][500];
float NOCSGBMResultinvalidPoint[maxCombinations][numOfCensusTypes][500];
float NOCSGBMResultwithinErrorHalf[maxCombinations][numOfCensusTypes][500];

float OCCactualExFastAverage = 0;
float NOCactualExFastAverage = 0;
int halfImage = 1;
float maxDisp = 255/(8*(halfImage+1));
int getImages = 0;
ofstream myfile;
float timeAverage =0;
float maxfx = 0.0 ;



/* For comparison, change the ground truth image for each function(occGt, nocGt, sparseoccGt .. ) indivisually as sometimes
differnt images are provided for occ and noc. */
int main(int argc, char** argv) {
	try {
		// Stereo matching parameters
		double uniqueness = 0.7;
		int leftRightStep = 2;
		
		// Feature detection parameters
		double adaptivity = 1.0;
		int minThreshold = 10;

 		char* calibFile = NULL;
		// int count=0;
		string LeftpathName = "/home/piyush/Documents/NTU Internship/stereo matching/SparseSetero Code/Baselines/training/";
		string RightpathName ="/home/piyush/Documents/NTU Internship/stereo matching/SparseSetero Code/Baselines/training/";
		 // namedWindow("CheckWindow",1);
		 // setMouseCallback("CheckWindow", CallBackFunc, NULL);
		 string fileType = ".png";	
		 int lexer;
		 int combination = 0;
		 int edgeNo = 0 ;
		 float matchesAfterDense = 0 ;
		 int inputImageSize = 0 ; 
		 float maxDisparityInput= 200;
		// Read input images
		for(lexer = 0 ; lexer <1;lexer++)
		 {
		 	cout<<"Lexer = "<<lexer<<endl;
		 	string filename = getFileName(lexer);

		 	cv::Mat_<unsigned char> leftImg, rightImg;	/*#*/
		 	Mat_<unsigned char> leftSGBMImage;
			Mat_<unsigned char> rightSGBMImage ;
			cv::Mat leftColoredImg,rightColoredImg;		/*#*/
			leftImg = imread("im0"+fileType, CV_LOAD_IMAGE_GRAYSCALE);
			leftSGBMImage = leftImg;

			rightImg = imread("im1"+fileType, CV_LOAD_IMAGE_GRAYSCALE);
			rightSGBMImage = rightImg;
			leftColoredImg  = imread("im0"+fileType, CV_LOAD_IMAGE_COLOR);
			rightColoredImg = imread("im1"+fileType, CV_LOAD_IMAGE_COLOR);
			//imshow("leftImg",leftColoredImg);
			if(halfImage == 1 && sparse ==0)
			{
				cv::resize(leftImg,leftImg,Size(leftImg.cols/2, leftImg.rows/2));
				cv::resize(leftColoredImg,leftColoredImg,Size(leftColoredImg.cols/2, leftColoredImg.rows/2));
				cv::resize(rightColoredImg,rightColoredImg,Size(rightColoredImg.cols/2, rightColoredImg.rows/2));
				cv::resize(rightImg,rightImg,Size(rightImg.cols/2, rightImg.rows/2));

			}
			// imshow("leftImg2",leftColoredImg);
			// waitKey();

			/* Initialize superPixelsoncee*/
		    	int prior = 2;
		   	 	bool double_step = true;
		    	int num_superpixels = 200;
		    	int num_levels = 3;
		    	int num_histogram_bins = 3;

		    	Ptr<SuperpixelSEEDS> seeds;
		    	Ptr<SuperpixelSEEDS> seeds0;

		    	int width, height;
		    	width = leftImg.size().width;
		        height = leftImg.size().height;
		        seeds = createSuperpixelSEEDS(width, height, leftColoredImg.channels(), num_superpixels, num_levels,
		        							  prior, num_histogram_bins, double_step);
		        seeds0 =  createSuperpixelSEEDS(width, height, leftColoredImg.channels(), num_superpixels, num_levels,
        							  prior, num_histogram_bins, double_step);
		    /* No reinitialization required */ 
		for(int censusWinType=3;censusWinType<numOfCensusTypes;censusWinType++)
		{ 	
			edgeNo = censusWinType+1;


			if(leftImg.data == NULL || rightImg.data == NULL)
				throw sparsestereo::Exception("Unable to open input images!");

		// Load rectification data
			StereoRectification* rectification = NULL;
				if(calibFile != NULL)
					rectification = new StereoRectification(CalibrationResult(calibFile));
		
		// The stereo matcher. SSE Optimized implementation is only available for a 5x5 window
			SparseStereo<CensusWindow<5>, long double> stereo(maxDisparityInput, 1, uniqueness,
				rectification, false, false, leftRightStep);
		
		vector<SparseMatch> correspondences;
		
		// Objects for storing final and intermediate results
		cv::Mat_<char> charLeft(leftImg.rows, leftImg.cols),		/*#*/
			charRight(rightImg.rows, rightImg.cols);
		Mat_<unsigned int> censusLeft(leftImg.rows, leftImg.cols),
			censusRight(rightImg.rows, rightImg.cols);
		vector<KeyPoint> keypointsLeft, keypointsRight;
		Mat leftLabels, leftMask, rightLabels, rightMask;
		Mat finalDisparityMap;
		ptime lastTime = microsec_clock::local_time();
		
			// Featuredetection. This part can be parallelized with OMP
			#pragma omp parallel sections default(shared) num_threads(2)
			{
				#pragma omp section
				{

					ImageConversion::unsignedToSigned(leftImg, &charLeft);
					Census::transform5x5(charLeft, &censusLeft, edgeNo);

					keypointsLeft.clear();
					keypointsLeft= getKeyPoints(leftColoredImg,leftLabels, leftMask, lexer, seeds);
				//	leftFeatureDetector->detect(leftImg, keypointsLeft);
				}
				#pragma omp section
				{
					ImageConversion::unsignedToSigned(rightImg, &charRight);
					Census::transform5x5(charRight, &censusRight,edgeNo);
					keypointsRight.clear();
					keypointsRight= getKeyPoints(rightColoredImg, rightLabels, rightMask, lexer, seeds0);
					//rightFeatureDetector->detect(rightImg, keypointsRight);
				}
			}

			correspondences.clear();

			stereo.match(censusLeft, censusRight, keypointsLeft, keypointsRight, &correspondences, combination);
			//cout<<"Total Number of matches after dense consistency check = "<<correspondences.size()<<endl;

			matchesAfterDense += correspondences.size() ;
			inputImageSize += leftColoredImg.rows*leftColoredImg.cols;
			Mat disparityImage(leftImg.rows, leftImg.cols, CV_32F, (float)0); /*#*/
			
			
			for(int a = 0 ; a<correspondences.size();a++)
			{	
				disparityImage.at<float>(correspondences.at(a).imgLeft->pt.y,correspondences.at(a).imgLeft->pt.x) = (correspondences.at(a).disparity());									
			}

			if(sparse == 0){
				sparseToDenseConversion(leftColoredImg,rightColoredImg, disparityImage,leftLabels,leftMask,correspondences,edgeNo);
				
				finalDisparityMap = postProcessingAndComparing(disparityImage, leftColoredImg);


				if(halfImage == 1)
					resize(finalDisparityMap,finalDisparityMap,Size(2*finalDisparityMap.cols, 2*finalDisparityMap.rows));
			
			if(getImages == 1)
			{
				  vector<int> compression_params;
				  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
				  compression_params.push_back(9);    
				  if(censusWinType == 7)
					{
				     cv::normalize(finalDisparityMap, finalDisparityMap, 0, 255, NORM_MINMAX, CV_32FC1);
				   	 imwrite("/home/piyush/Downloads/original exFast/Image outputs/ShelvesA.png", finalDisparityMap, compression_params);
					}
			}
		

				//finalDisparityMap = disparityImage;
			}
			else{
				
				finalDisparityMap = disparityImage;
			}
		    time_duration elapsed = (microsec_clock::local_time() - lastTime);
			timeAverage +=  elapsed.total_microseconds()/1.0e6;

	
		
		ptime lastTime2 = microsec_clock::local_time();
		
		Mat SGBMDisp = disparityMapOfSGBM(leftSGBMImage,rightSGBMImage); 	//to read the values use the following function(for cv_16U type values)	 (float)disp8.at<short>(i,j)/16.0f
		time_duration elapsed2 = (microsec_clock::local_time() - lastTime2);
			cout << "Time for SGBM matching: " << elapsed2.total_microseconds()/1.0e6 << "s" << endl;
	    	//timeAverage +=  elapsed2.total_microseconds()/1.0e6;
		comparingOfDisparityMap(finalDisparityMap, correspondences,lexer,LeftpathName,RightpathName,fileType,filename,censusWinType,combination);

		SgbmGTCompare(correspondences,lexer,fileType,filename,SGBMDisp,censusWinType,combination);

		if(rectification != NULL)
			delete rectification;
		}

		
			
		}
	    printToAFile(lexer);
		cout<<"Average time = " << timeAverage/(lexer*7)<<endl;
		cout<<"Average Size = " << inputImageSize/(lexer*7)<<endl;
		cout<<"Average Matches = "<<matchesAfterDense/(lexer*7.0)<<endl;
		cout<<"Average Sparse To Dense Percentage ="<<matchesAfterDense*100/inputImageSize<<endl;
	}
	catch (const std::exception& e) {
		cerr << "Fatal exception: " << e.what();
		return 1;
	}
	myfile.close();
}

Mat disparityMapOfSGBM(cv::Mat_<unsigned char> leftImg,cv::Mat_<unsigned char>  rightImg)
{
	int SADWindowSize=3, numberOfDisparities=80;
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
	Mat img1 = leftImg.clone();
    Mat img2 = rightImg.clone();

    sgbm->setPreFilterCap(63);
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);

    int cn = img1.channels();

    sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1); 
    sgbm->setMode(StereoSGBM::MODE_SGBM);

    Mat disp, disp8;
    sgbm->compute(img1, img2, disp);
    disp.convertTo(disp8, CV_16U);
    return disp8;
}

void sparseToDenseConversion(Mat leftColoredImg, Mat rightColoredImg, Mat& disparityImage,Mat labels,Mat leftMask, vector<SparseMatch> correspondences, int edgeNo)
{
	/*Store the indexes and groups first*/
	RNG rng(12345);

	int indexHash[1000];							//store the label and the index of the vector array to which all the points 
	memset(indexHash,-1,1000*sizeof(int));												//having the label correspond to 
	vector<labelOfPoints> boundarySuperPixel;		//store the point as well as the label of the point
	int index = -1;
	int boundaryIndex = 0 ;
	labelOfPoints temp;
	Mat leftColorLab, rightColorLab;/*#*/
	cvtColor(leftColoredImg, leftColorLab, CV_BGR2Lab);


				

	for(int i=0;i<correspondences.size();i++)	
	{	
		index = -1;
		temp.label = -1;
		temp.points.clear();
		index = labels.at<unsigned int>(correspondences.at(i).imgLeft->pt.y,correspondences.at(i).imgLeft->pt.x);
		
		if(indexHash[index] ==-1)
		{
			indexHash[index] =boundaryIndex;
			temp.label = index;
			temp.points.push_back(Point(correspondences.at(i).imgLeft->pt.x,correspondences.at(i).imgLeft->pt.y));
			boundarySuperPixel.push_back(temp) ;
			
			boundaryIndex++;
		}
		else
		{
			if(boundarySuperPixel.at(indexHash[index]).label == index){
					boundarySuperPixel.at(indexHash[index]).points.push_back(Point(correspondences.at(i).imgLeft->pt.x,correspondences.at(i).imgLeft->pt.y)) ;
			}
		}
	}
	
	


	/*FOR COSINE AND ADAPTIVE INTERPOLATION*/

	/* Indexes of non filled points even after interpolation */
	vector<Point2f> invalidPoints; 

	

	int rows = disparityImage.rows;
   for(int i =0 ; i<rows;i++)	//y
		for(int j=0;j<disparityImage.cols;j++)	//x
		{
			if(disparityImage.at<float>(i,j)== 0 && indexHash[labels.at<unsigned int>(i,j)] != -1)
	 		{
	 			int arrayIndex = indexHash[labels.at<unsigned int>(i,j)];
	 			int tempLoopSize = boundarySuperPixel.at(arrayIndex).points.size();
	 			
	 			int diffLeft = rows/10 ; 
	 			int diffRight = rows/10 ; 

	 			int leftPointIndex = -1; 
	 			int rightPointIndex = -1;
	 			for(int k = 0 ; k < tempLoopSize;k++)
	 			{
	 				if(abs(boundarySuperPixel.at(arrayIndex).points.at(k).y - i) < diffLeft && (boundarySuperPixel.at(arrayIndex).points.at(k).x -j) >0)	//right index
	 				{
	 					rightPointIndex = k ;
	 					diffLeft = abs(boundarySuperPixel.at(arrayIndex).points.at(k).y - i) ; 
	 				}

	 				if(abs(boundarySuperPixel.at(arrayIndex).points.at(k).y - i) < diffRight && (boundarySuperPixel.at(arrayIndex).points.at(k).x -j) <0)	//left index
	 				{
	 					leftPointIndex = k ;
	 					diffRight = abs(boundarySuperPixel.at(arrayIndex).points.at(k).y - i) ; 
	 				}
	 			}

	 			if(leftPointIndex == -1 || rightPointIndex == -1)
	 			{
	 					 float Yc = 12, Yp = 5,gain =1;

						int tempLoopSize = boundarySuperPixel.at(indexHash[labels.at<unsigned int>(i,j)]).points.size();
						float correlationDenum=0;
						for(int k=0 ; k<tempLoopSize ;k++)
						{
							/*Compute Weight for the pixel*/
							int y = boundarySuperPixel.at(indexHash[labels.at<unsigned int>(i,j)]).points.at(k).y;
							int x =  boundarySuperPixel.at(indexHash[labels.at<unsigned int>(i,j)]).points.at(k).x ;
							float colorDiff = pow( pow((short)leftColorLab.at<cv::Vec3b>(i ,j)[0] - (short)leftColorLab.at<cv::Vec3b>(y,x)[0],2)
											+ pow((short)leftColorLab.at<cv::Vec3b>(i ,j)[1] - (short)leftColorLab.at<cv::Vec3b>(y,x)[1],2) 
											+ pow((short)leftColorLab.at<cv::Vec3b>(i ,j)[2] - (short)leftColorLab.at<cv::Vec3b>(y,x)[2],2),0.5);
							
							float spatialDiff = pow(pow(i-y,2)+pow(j-x,2),0.5);
							 float weight =  gain*exp(-((colorDiff/Yc)+(spatialDiff/Yp)));

							correlationDenum += weight;

							disparityImage.at<float>(i,j) +=weight*(float)disparityImage.at<float>(y,x);
						}
						disparityImage.at<float>(i,j) = disparityImage.at<float>(i,j)/correlationDenum ;
						
	 			}
	 			else
	 			{
		 			double mu =(boundarySuperPixel.at(arrayIndex).points.at(rightPointIndex).x - j)/(boundarySuperPixel.at(arrayIndex).points.at(rightPointIndex).x -boundarySuperPixel.at(arrayIndex).points.at(leftPointIndex).x );
		 			double mu2 = (1-cos(mu*PI))/2;
		 			double disp2 = disparityImage.at<float>(boundarySuperPixel.at(arrayIndex).points.at(rightPointIndex).y,boundarySuperPixel.at(arrayIndex).points.at(rightPointIndex).x);
		 			double disp1 = disparityImage.at<float>(boundarySuperPixel.at(arrayIndex).points.at(leftPointIndex).y,boundarySuperPixel.at(arrayIndex).points.at(leftPointIndex).x);

		 			disparityImage.at<float>(i,j) = disp2*(1-mu2) + disp1*mu2;
		 		}

		 		
	 		}
	 		if(disparityImage.at<float>(i,j)== 0)
		 			invalidPoints.push_back(Point2f(j,i));
		}
	
		if(getImages ==1 )
		{

		    vector<int> compression_params;
				    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
				    compression_params.push_back(9);    
				    if(edgeNo == 5){
				    	 cv::normalize(disparityImage, disparityImage, 0, 255, NORM_MINMAX, CV_32FC1);
				    	imwrite("/home/piyush/Downloads/original exFast/Image outputs/Shelves.png", disparityImage, compression_params);
				    }
		}


		/* Take care of the rest of the invalid pixels */
		for(int i = 0 ; i<invalidPoints.size();i++)
		{
			for(int j = invalidPoints.at(i).x; j<disparityImage.cols;j++)
			{
				if(disparityImage.at<float>(invalidPoints.at(i).y,j) ==0)
					continue;
				else
				{
					disparityImage.at<float>(invalidPoints.at(i).y,invalidPoints.at(i).x) = disparityImage.at<float>(invalidPoints.at(i).y,j);
					break;
				}
			}
		}

}


Mat postProcessingAndComparing(Mat disp/*#*/, Mat leftImage)
{
	
/*Postprocessing part for The disparity based on the sgbm post processing code*/
    medianBlur(disp, disp, 3);
	Mat buffer;
	Mat image = filterSpeckles1(disp, 0, 160, 3, buffer);

	for(int i=0; i<image.rows;i++)
		for(int j=0;j<image.cols;j++)
		{
			if(image.at<float>(i,j)==0)
			{
				int k=j ; 
				while(image.at<float>(i,j) ==0 && k<image.cols)
				{
					k++;
					image.at<float>(i,j)=image.at<float>(i,k);
				}
			}
		}
	return image;
}
Mat filterSpeckles1( InputOutputArray _img, double _newval, int maxSpeckleSize, double _maxDiff, InputOutputArray __buf)
{

    Mat img = _img.getMat();
    int type = img.type();
    Mat temp, &_buf = __buf.needed() ? __buf.getMatRef() : temp;
    //  CV_Assert( type == CV_8UC1 || type == CV_16SC1 );

    int newVal = cvRound(_newval), maxDiff = cvRound(_maxDiff);

   // CV_IPP_RUN(IPP_VERSION_X100 >= 810 && !__buf.needed() && (type == CV_8UC1 || type == CV_16SC1), ipp_filterSpeckles1(img, maxSpeckleSize, newVal, maxDiff));

  //  if (type == CV_8UC1)
    
    filterSpeckles1Impl(img, newVal, maxSpeckleSize, maxDiff, _buf);
    return img;
  //  comparisonImageOutput(img,correspondences2);

    // else
    //     filterSpeckles1Impl< short>(img, newVal, maxSpeckleSize, maxDiff, _buf);
}

void filterSpeckles1Impl(cv::Mat& img, int newVal, int maxSpeckleSize, int maxDiff, cv::Mat& _buf)
{
    using namespace cv;

    int width = img.cols, height = img.rows, npixels = width*height;
    size_t bufSize = npixels*(int)(sizeof(Point2i) + sizeof(int) + sizeof(float));
    if( !_buf.isContinuous() || _buf.empty() || _buf.cols*_buf.rows*_buf.elemSize() < bufSize )
        _buf.create(1, (int)bufSize, CV_32F);


    //might not be the correct way to do this >>>> Check again
    uchar* buf = _buf.ptr();
    int i, j, dstep = (int)(img.step/sizeof(float));
    int* labels = (int*)buf;
    buf += npixels*sizeof(labels[0]);
    Point2i* wbuf = (Point2i*)buf;
    buf += npixels*sizeof(wbuf[0]);
    float* rtype = (float*)buf;
    int curlabel = 0;

    // clear out label assignments
    memset(labels, 0, npixels*sizeof(labels[0]));

    for( i = 0; i < height; i++ )
    {
        float* ds = img.ptr<float>(i);
        int* ls = labels + width*i;

        for( j = 0; j < width; j++ )
        {
            if( ds[j] != newVal )   // not a bad disparity
            {
                if( ls[j] )     // has a label, check for bad label
                {
                    if( rtype[ls[j]] ) // small region, zero out disparity
                        ds[j] = (float)newVal;
                }
                // no label, assign and propagate
                else
                {
                    Point2i* ws = wbuf; // initialize wavefront
                    Point2i p((short)j, (short)i);  // current pixel
                    curlabel++; // next label
                    int count = 0;  // current region size
                    ls[j] = curlabel;

                    // wavefront propagation
                    while( ws >= wbuf ) // wavefront not empty
                    {
                        count++;
                        // put neighbors onto wavefront
                        float* dpp = &img.at<float>(p.y, p.x);
                        float dp = *dpp;
                        int* lpp = labels + width*p.y + p.x;

                        if( p.y < height-1 && !lpp[+width] && dpp[+dstep] != newVal && std::abs(dp - dpp[+dstep]) <= maxDiff )
                        {
                            lpp[+width] = curlabel;
                            *ws++ = Point2i(p.x, p.y+1);
                        }

                        if( p.y > 0 && !lpp[-width] && dpp[-dstep] != newVal && std::abs(dp - dpp[-dstep]) <= maxDiff )
                        {
                            lpp[-width] = curlabel;
                            *ws++ = Point2i(p.x, p.y-1);
                        }

                        if( p.x < width-1 && !lpp[+1] && dpp[+1] != newVal && std::abs(dp - dpp[+1]) <= maxDiff )
                        {
                            lpp[+1] = curlabel;
                            *ws++ = Point2i(p.x+1, p.y);
                        }

                        if( p.x > 0 && !lpp[-1] && dpp[-1] != newVal && std::abs(dp - dpp[-1]) <= maxDiff )
                        {
                            lpp[-1] = curlabel;
                            *ws++ = Point2i(p.x-1, p.y);
                        }

                        // pop most recent and propagate
                        // NB: could try least recent, maybe better convergence
                        p = *--ws;
                    }

                    // assign label type
                    if( count <= maxSpeckleSize )   // speckle region
                    {
                        rtype[ls[j]] = 1;   // small region label
                        ds[j] = (float)newVal;
                    }
                    else
                        rtype[ls[j]] = 0;   // large region label
                }
            }
        }
    }
}



string getFileName(int lexer)
{
		string filename =""; 

		int count = lexer;
		 int decimalPlaces = 0 ;
		 int decimalOuput = -1;
		 while(decimalOuput!=0)
		 {
		 	decimalOuput = count/10;
		 	count = count/10 ;
		 	decimalPlaces++ ; 
		 }
		 for(int i = 0 ;i<6-decimalPlaces;i++)
		 {
		 	filename = filename +"0";
		 } 
		 stringstream ss;
		 ss << lexer;
		 string fileNumber = ss.str();

		 filename = filename + fileNumber +"_10";
		 return filename;
}


void comparingOfDisparityMap(Mat disp,vector<SparseMatch> correspondences, int lexer,string LeftpathName, string RightpathName, string fileType,string filename,  int censusWinType, int combination)
{
	if(sparse == 0)
	{
		occGTCompare(disp,correspondences,lexer,fileType,filename,censusWinType,combination);
		nonoccGTCompare(disp,correspondences,lexer,fileType,filename,censusWinType,combination);
	}
	else if(sparse == 1)
	{
		occGTCompareSparse(correspondences,lexer,fileType,filename,censusWinType,combination);
		nonoccGTCompareSparse(correspondences,lexer,fileType,filename,censusWinType,combination);
	}

}



void nonoccGTCompare(Mat disp, vector<SparseMatch> correspondences, int lexer, string fileType, string filename, int censusWinType, int combination)
{
	//string fileDirectory = "/home/piyush/Documents/NTU Internship/stereo matching/SparseSetero Code/Baselines/training/disp_noc/";
	Mat GT = imread("disp2.png", -1);/*#*/
	int totalNumberOfMatchedFeatures = GT.cols*GT.rows;
	int withinError1= 0 ;
	int withinError2= 0 ; 
	int withinError3=0;
	int errorMoreThan3=0;
	int invalidPoint=0;
	int withinErrorHalf=0 ; 
	cout<<"Gt Row = "<<GT.rows<<endl;
	cout<<"Gt cols = "<<GT.cols<<endl;

	for(int i=0;i<GT.rows;i++)
	{
		for(int j=0;j<(int)GT.cols;j++)
		{
			if((int)(GT.at<uchar>(i,j)) ==0)
				invalidPoint++;
			else
			{
				float x = abs((float)disp.at<float>(i,j)-(float)((int)(GT.at<uchar>(i,j)))*(float)(maxDisp/255));				//cout<<"Disparity = "<<(float)disp.at<float>(i,j)<<" , GT = "<<((int)(GT.at<uchar>(i,j)))<<endl;
				if(x<=0.5)
				{
						withinErrorHalf++;
				}
				else if(x <=1  )
					withinError1++;
				else if(x<=2 )
					withinError2++ ; 
				else if(x<=3 )
					withinError3++;
				else if(x>3)
					{
						//error.push_back(cv::KeyPoint(correspondences[indexes[i]].rectLeft.x, correspondences[indexes[i]].rectLeft.y));
						errorMoreThan3++;
					}

			
			
			}
		}
	}

	 NOCcannyResultwithinError1[combination][censusWinType][lexer] = withinError1;
 	 NOCcannyResultwithinError2[combination][censusWinType][lexer] = withinError2;
	 NOCcannyResultwithinError3[combination][censusWinType][lexer] = withinError3;
	 NOCcannyResulterrorMoreThan3[combination][censusWinType][lexer] = errorMoreThan3;
	 NOCcannyResultNumberOfFeatures[combination][censusWinType][lexer] = totalNumberOfMatchedFeatures;
 	 NOCcannyResultinvalidPoint[combination][censusWinType][lexer] = invalidPoint;
 	 NOCcannyResultwithinErrorHalf[combination][censusWinType][lexer] = withinErrorHalf;
}

void occGTCompare(Mat disp, vector<SparseMatch> correspondences, int lexer, string fileType,string filename,int censusWinType,int combination)
{
	string fileDirectory = "/home/piyush/Documents/NTU Internship/stereo matching/SparseSetero Code/Baselines/training/disp_occ/";
	Mat GT = imread("disp2.png", -1);
	//string fileDirectory2 = "/home/piyush/Documents/NTU Internship/stereo matching/SparseSetero Code/Baselines/training/colored_0/";
	//Mat image = imread("im0.png", CV_LOAD_IMAGE_COLOR);
	//Mat image = Mat::zeros(GT.size(),CV_8UC3);

	int totalNumberOfMatchedFeatures = GT.cols*GT.rows;
	int withinError1= 0 ;
	int withinError2= 0 ; 
	int withinError3=0;
	int errorMoreThan3=0;
	int invalidPoint=0;
	int withinErrorHalf=0 ; 

	 	// ofstream  debugFile;
	 	//  stringstream ss;
		 // ss << combination;
		 // string combinationNum = ss.str();
	 //debugFile.open ("Combination"+combinationNum+".txt");
	//Mat blank = Mat::zeros(image.size(),CV_8UC3);
	for(int i=0;i<GT.rows;i++)
	{
			for(int j=0;j<(int)GT.cols;j++)
			{
			//	disparityFile<<"Point ("<<correspondences[i].imgLeft->pt.y<<" "<<correspondences[i].imgLeft->pt.x<<")"<<"="<<(float)correspondences[i].disparity()<<endl;

				if((int)(GT.at<uchar>(i,j))==0 )
					invalidPoint++;
				else
				{
						float x = abs((float)disp.at<float>(i,j)-(float)((int)(GT.at<uchar>(i,j)))*(float)(maxDisp/255));
						//cout<<"Disparity = "<<(float)disp.at<float>(i,j)<<" , gt = "<<((int)(int)(GT.at<uchar>(i,j)))*(float)(maxDisp/255<<en)dl;
						if(x<=0.5){
							// circle(image,Point(j,i),2,Scalar(255,255,255),1,8,0);
							withinErrorHalf++;
						}
						else if(x <=1 )
							withinError1++;
						else if(x<=2 )
							withinError2++ ; 
						else if(x<=3 )
							withinError3++;
						else if(x>3){
							// circle(image,Point(j,i),2,Scalar(0,0,255),1,8,0);
							errorMoreThan3++;
						}

						// if(x>3)
						// {
						// 	//image.at<unsigned int>(i,j) = ; 
						// }
						// else{

						// 	image.at<unsigned int>(i,j) = image.at<unsigned int>(i,j)*0.8;
						// }


					}
							//cout<<"Line 495"<<endl;

			}
	}
	// cout<<"Hey"<<endl;

	 // imshow("Image",image);
	 // waitKey();

	 OCCcannyResultwithinError1[combination][censusWinType][lexer] = withinError1;
	 OCCcannyResultwithinError2[combination][censusWinType][lexer] = withinError2;
	 OCCcannyResultwithinError3[combination][censusWinType][lexer] = withinError3;
	 OCCcannyResulterrorMoreThan3[combination][censusWinType][lexer] = errorMoreThan3;
	 OCCcannyResultNumberOfFeatures[combination][censusWinType][lexer] = totalNumberOfMatchedFeatures;
	 OCCcannyResultinvalidPoint[combination][censusWinType][lexer] = invalidPoint;
	 OCCcannyResultwithinErrorHalf[combination][censusWinType][lexer] = withinErrorHalf;
	 	 	 
}



void nonoccGTCompareSparse( vector<SparseMatch> correspondences, int lexer, string fileType, string filename, int censusWinType, int combination)
{
	string fileDirectory = "/home/piyush/Documents/NTU Internship/stereo matching/SparseSetero Code/Baselines/training/disp_noc/";
	Mat GT = imread("disp2.png", -1);
	int totalNumberOfMatchedFeatures = correspondences.size();
	int withinError1= 0 ;
	int withinError2= 0 ; 
	int withinError3=0;
	int errorMoreThan3=0;
	int invalidPoint=0;
	int withinErrorHalf=0 ; 


	for(int i =0  ;i<correspondences.size();i++)
	{

		if((float)(GT.at<uchar>(correspondences[i].rectLeft.y,correspondences[i].rectLeft.x) ==0))
			invalidPoint++;
		else
		{
				float x = abs( correspondences.at(i).disparity() - (float)((int)GT.at<uchar>(correspondences[i].rectLeft.y,correspondences[i].rectLeft.x)*(float)(maxDisp/255.0)));
				if(x<=0.5)
				{
					withinErrorHalf++;
					
				}
			else if(x <=1)
				withinError1++;
			else if(x<=2 )
				withinError2++ ; 
			else if(x<=3 )
				withinError3++;
			else if(x>3)
				{
					errorMoreThan3++;
				}

		
		
		}
	}

	 NOCcannyResultwithinError1[combination][censusWinType][lexer] = withinError1;
 	 NOCcannyResultwithinError2[combination][censusWinType][lexer] = withinError2;
	 NOCcannyResultwithinError3[combination][censusWinType][lexer] = withinError3;
	 NOCcannyResulterrorMoreThan3[combination][censusWinType][lexer] = errorMoreThan3;
	 NOCcannyResultNumberOfFeatures[combination][censusWinType][lexer] = totalNumberOfMatchedFeatures;
 	 NOCcannyResultinvalidPoint[combination][censusWinType][lexer] = invalidPoint;
 	 NOCcannyResultwithinErrorHalf[combination][censusWinType][lexer] = withinErrorHalf;
}

void occGTCompareSparse( vector<SparseMatch> correspondences, int lexer, string fileType,string filename,int censusWinType,int combination)
{
	string fileDirectory = "/home/piyush/Documents/NTU Internship/stereo matching/SparseSetero Code/Baselines/training/disp_occ/";
	Mat GT = imread("disp2.png", -1);
	//string fileDirectory2 = "/home/piyush/Documents/NTU Internship/stereo matching/SparseSetero Code/Baselines/training/colored_0/";
	//Mat image = imread(fileDirectory2+filename+fileType, CV_LOAD_IMAGE_COLOR);
	//Mat image =imread("im0.png",-1); //To show the error pixels in image
	int totalNumberOfMatchedFeatures = correspondences.size();
	int withinError1= 0 ;
	int withinError2= 0 ; 
	int withinError3=0;
	int errorMoreThan3=0;
	int invalidPoint=0;
	int withinErrorHalf=0 ; 
	

	for(int i =0  ;i<correspondences.size();i++)
	{

		if((float)(GT.at<uchar>(correspondences[i].rectLeft.y,correspondences[i].rectLeft.x) ==0))
			invalidPoint++;
		else
		{
				float x = abs( correspondences.at(i).disparity() - (float)((int)GT.at<uchar>(correspondences[i].rectLeft.y,correspondences[i].rectLeft.x)*(float)(maxDisp/255.0)));
				if(x<=0.5){
					 // circle(image,Point(correspondences[i].rectLeft.x,correspondences[i].rectLeft.y),1,Scalar(255,255,255),1,8,0);	

					withinErrorHalf++;
				}
				else if(x <=1 && x>0.5 ){
					// circle(image,Point(correspondences[i].rectLeft.x,correspondences[i].rectLeft.y),1,Scalar(0,255,255),1,8,0);	

					withinError1++;
				}
				else if(x<=2 && x>1){
					// circle(image,Point(correspondences[i].rectLeft.x,correspondences[i].rectLeft.y),1,Scalar(0,255,0),1,8,0);	

					withinError2++ ; 
				}
				else if(x<=3 && x>2){
					// circle(image,Point(correspondences[i].rectLeft.x,correspondences[i].rectLeft.y),1,Scalar(255,0,0),1,8,0);	

					withinError3++;
				}
				else if(x>3){
					 circle(image,Point(correspondences[i].rectLeft.x,correspondences[i].rectLeft.y),1,Scalar(0,0,255),1,8,0);	

					errorMoreThan3++;
				}

				// if(x>3)
				// 	circle(image,Point(correspondences[i].rectLeft.x,correspondences[i].rectLeft.y),1,Scalar(0,0,255),1,8,0);	
				// else
				// 	circle(image,Point(correspondences[i].rectLeft.x,correspondences[i].rectLeft.y),1,Scalar(255,255,255),1,8,0);	

			}

	}
	 OCCcannyResultwithinError1[combination][censusWinType][lexer] = withinError1;
	OCCcannyResultwithinError2[combination][censusWinType][lexer] = withinError2;
	OCCcannyResultwithinError3[combination][censusWinType][lexer] = withinError3;
	OCCcannyResulterrorMoreThan3[combination][censusWinType][lexer] = errorMoreThan3;
	OCCcannyResultNumberOfFeatures[combination][censusWinType][lexer] = totalNumberOfMatchedFeatures;
	OCCcannyResultinvalidPoint[combination][censusWinType][lexer] = invalidPoint;
	OCCcannyResultwithinErrorHalf[combination][censusWinType][lexer] = withinErrorHalf;
 	
 }


void printToAFile(int lexer)
{
	//to a matlab file 
	ofstream matlabFile;
	matlabFile.open ("Plot.m");
	float OCCaverageForCannyEdgeWithinError1[maxCombinations][numOfCensusTypes] ;
	float OCCaverageForCannyEdgeWithinError2[maxCombinations][numOfCensusTypes]  ;
	float OCCaverageForCannyEdgeWithinError3[maxCombinations][numOfCensusTypes]  ;
	float OCCaverageForCannyEdgeErrorMoreThan3[maxCombinations][numOfCensusTypes] ;
	float OCCaverageForCannyEdgeWithinErrorHalf[maxCombinations][numOfCensusTypes] ;

	float NOCaverageForCannyEdgeWithinError1[maxCombinations][numOfCensusTypes] ;
	float NOCaverageForCannyEdgeWithinError2[maxCombinations][numOfCensusTypes]  ;
	float NOCaverageForCannyEdgeWithinError3[maxCombinations][numOfCensusTypes] ;
	float NOCaverageForCannyEdgeErrorMoreThan3[maxCombinations][numOfCensusTypes] ;
	float NOCaverageForCannyEdgeWithinErrorHalf[maxCombinations][numOfCensusTypes] ;


	float OCCaverageForSGBMWithinError1[maxCombinations][numOfCensusTypes] ; 
	float OCCaverageForSGBMWithinError2[maxCombinations][numOfCensusTypes] ; 
	float OCCaverageForSGBMWithinError3[maxCombinations][numOfCensusTypes] ; 
	float OCCaverageForSGBMErrorMoreThan3[maxCombinations][numOfCensusTypes] ; 
	float OCCaverageForSGBMWithinErrorHalf[maxCombinations][numOfCensusTypes] ; 


	float NOCaverageForSGBMWithinError1[maxCombinations][numOfCensusTypes] ; 
	float NOCaverageForSGBMWithinError2[maxCombinations][numOfCensusTypes] ; 
	float NOCaverageForSGBMWithinError3[maxCombinations][numOfCensusTypes] ; 
	float NOCaverageForSGBMErrorMoreThan3[maxCombinations][numOfCensusTypes] ; 
	float NOCaverageForSGBMWithinErrorHalf[maxCombinations][numOfCensusTypes] ; 
for(int k =0 ; k<maxCombinations;k++)
{

			for(int i=0;i<numOfCensusTypes;i++)
			{	
			 OCCaverageForCannyEdgeWithinError1[k][i] =0;
			 OCCaverageForCannyEdgeWithinError2[k][i]  =0;
			 OCCaverageForCannyEdgeWithinError3[k][i]  =0;
			 OCCaverageForCannyEdgeErrorMoreThan3[k][i] =0;
		     OCCaverageForCannyEdgeWithinErrorHalf[k][i] =0;

			 NOCaverageForCannyEdgeWithinError1[k][i] =0;
			 NOCaverageForCannyEdgeWithinError2[k][i]  =0;
			 NOCaverageForCannyEdgeWithinError3[k][i] =0;
			 NOCaverageForCannyEdgeErrorMoreThan3[k][i] =0;
			 NOCaverageForCannyEdgeWithinErrorHalf[k][i] =0;


			 OCCaverageForSGBMWithinError1[k][i] =0; 
			 OCCaverageForSGBMWithinError2[k][i] =0; 
			 OCCaverageForSGBMWithinError3[k][i] =0; 
			 OCCaverageForSGBMErrorMoreThan3[k][i] =0; 
			 OCCaverageForSGBMWithinErrorHalf[k][i]=0 ; 


			 NOCaverageForSGBMWithinError1[k][i] =0; 
			 NOCaverageForSGBMWithinError2[k][i] =0; 
			 NOCaverageForSGBMWithinError3[k][i] =0; 
			 NOCaverageForSGBMErrorMoreThan3[k][i] =0;
			 NOCaverageForSGBMWithinErrorHalf[k][i]=0 ; 
	

				for(int j=0;j<lexer;j++)
				{
					

					 OCCaverageForCannyEdgeWithinError1[k][i]    +=OCCcannyResultwithinError1[k][i][j]/(OCCcannyResultNumberOfFeatures[k][i][j]    - OCCcannyResultinvalidPoint[k][i][j]) ;
					 OCCaverageForCannyEdgeWithinError2[k][i]    +=OCCcannyResultwithinError2[k][i][j]/(OCCcannyResultNumberOfFeatures[k][i][j]    - OCCcannyResultinvalidPoint[k][i][j]) ;
					 OCCaverageForCannyEdgeWithinError3[k][i]    +=OCCcannyResultwithinError3[k][i][j]/(OCCcannyResultNumberOfFeatures[k][i][j]    - OCCcannyResultinvalidPoint[k][i][j]) ;
					 OCCaverageForCannyEdgeErrorMoreThan3[k][i]  +=OCCcannyResulterrorMoreThan3[k][i][j]/(OCCcannyResultNumberOfFeatures[k][i][j]  - OCCcannyResultinvalidPoint[k][i][j]) ;
				     OCCaverageForCannyEdgeWithinErrorHalf[k][i] +=OCCcannyResultwithinErrorHalf[k][i][j]/(OCCcannyResultNumberOfFeatures[k][i][j] - OCCcannyResultinvalidPoint[k][i][j]) ;

					 NOCaverageForCannyEdgeWithinError1[k][i]   +=NOCcannyResultwithinError1[k][i][j]/(NOCcannyResultNumberOfFeatures[k][i][j]   - NOCcannyResultinvalidPoint[k][i][j]) ; 
					 NOCaverageForCannyEdgeWithinError2[k][i]   +=NOCcannyResultwithinError2[k][i][j]/(NOCcannyResultNumberOfFeatures[k][i][j]   - NOCcannyResultinvalidPoint[k][i][j]) ; 
					 NOCaverageForCannyEdgeWithinError3[k][i]   +=NOCcannyResultwithinError3[k][i][j]/(NOCcannyResultNumberOfFeatures[k][i][j]   - NOCcannyResultinvalidPoint[k][i][j]) ; 
					 NOCaverageForCannyEdgeErrorMoreThan3[k][i] +=NOCcannyResulterrorMoreThan3[k][i][j] /(NOCcannyResultNumberOfFeatures[k][i][j]- NOCcannyResultinvalidPoint[k][i][j]) ; 
					 NOCaverageForCannyEdgeWithinErrorHalf[k][i]+=NOCcannyResultwithinErrorHalf[k][i][j]/(NOCcannyResultNumberOfFeatures[k][i][j]- NOCcannyResultinvalidPoint[k][i][j]) ; 

					 OCCaverageForSGBMWithinError1[k][i]   +=OCCSGBMResultwithinError1[k][i][j] /(OCCSGBMResultNumberOfFeatures[k][i][j]  -OCCSGBMResultinvalidPoint[k][i][j]); 
					 OCCaverageForSGBMWithinError2[k][i]   +=OCCSGBMResultwithinError2[k][i][j]/(OCCSGBMResultNumberOfFeatures[k][i][j]   -OCCSGBMResultinvalidPoint[k][i][j]); 
					 OCCaverageForSGBMWithinError3[k][i]   +=OCCSGBMResultwithinError3[k][i][j]/(OCCSGBMResultNumberOfFeatures[k][i][j]   -OCCSGBMResultinvalidPoint[k][i][j]); 
					 OCCaverageForSGBMErrorMoreThan3[k][i] +=OCCSGBMResulterrorMoreThan3[k][i][j]/(OCCSGBMResultNumberOfFeatures[k][i][j] -OCCSGBMResultinvalidPoint[k][i][j]); 
					 OCCaverageForSGBMWithinErrorHalf[k][i]+=OCCSGBMResultwithinErrorHalf[k][i][j]/(OCCSGBMResultNumberOfFeatures[k][i][j]-OCCSGBMResultinvalidPoint[k][i][j]);  

					 NOCaverageForSGBMWithinError1[k][i] +=NOCSGBMResultwithinError1[k][i][j]/(NOCSGBMResultNumberOfFeatures[k][i][j]-NOCSGBMResultinvalidPoint[k][i][j]); 
					 NOCaverageForSGBMWithinError2[k][i] +=NOCSGBMResultwithinError2[k][i][j]/(NOCSGBMResultNumberOfFeatures[k][i][j]-NOCSGBMResultinvalidPoint[k][i][j]); 
					 NOCaverageForSGBMWithinError3[k][i] +=NOCSGBMResultwithinError3[k][i][j]/(NOCSGBMResultNumberOfFeatures[k][i][j]-NOCSGBMResultinvalidPoint[k][i][j]); 
					 NOCaverageForSGBMErrorMoreThan3[k][i] +=NOCSGBMResulterrorMoreThan3[k][i][j]/(NOCSGBMResultNumberOfFeatures[k][i][j]-NOCSGBMResultinvalidPoint[k][i][j]);
					 NOCaverageForSGBMWithinErrorHalf[k][i] +=	NOCSGBMResultwithinErrorHalf[k][i][j]/(NOCSGBMResultNumberOfFeatures[k][i][j]-NOCSGBMResultinvalidPoint[k][i][j]); 





				}
			 OCCaverageForCannyEdgeWithinError1[k][i]  =( OCCaverageForCannyEdgeWithinError1[k][i]/lexer)*100;
			 OCCaverageForCannyEdgeWithinError2[k][i]  =( OCCaverageForCannyEdgeWithinError2[k][i]/lexer)*100;
			 OCCaverageForCannyEdgeWithinError3[k][i]  =(OCCaverageForCannyEdgeWithinError3[k][i]/lexer)*100;
			 OCCaverageForCannyEdgeErrorMoreThan3[k][i] =(OCCaverageForCannyEdgeErrorMoreThan3[k][i]/lexer)*100;
		     OCCaverageForCannyEdgeWithinErrorHalf[k][i] = (OCCaverageForCannyEdgeWithinErrorHalf[k][i]/lexer)*100;

			 NOCaverageForCannyEdgeWithinError1[k][i] =(NOCaverageForCannyEdgeWithinError1[k][i]/lexer)*100;
			 NOCaverageForCannyEdgeWithinError2[k][i]  =(NOCaverageForCannyEdgeWithinError2[k][i] /lexer)*100;
			 NOCaverageForCannyEdgeWithinError3[k][i] =(NOCaverageForCannyEdgeWithinError3[k][i]/lexer)*100;
			 NOCaverageForCannyEdgeErrorMoreThan3[k][i] =(NOCaverageForCannyEdgeErrorMoreThan3[k][i]/lexer)*100;
		     NOCaverageForCannyEdgeWithinErrorHalf[k][i] = (NOCaverageForCannyEdgeWithinErrorHalf[k][i]/lexer)*100;

			 OCCaverageForSGBMWithinError1[k][i] =(OCCaverageForSGBMWithinError1[k][i]/lexer)*100; 
			 OCCaverageForSGBMWithinError2[k][i] =( OCCaverageForSGBMWithinError2[k][i] /lexer)*100; 
			 OCCaverageForSGBMWithinError3[k][i] =( OCCaverageForSGBMWithinError3[k][i]/lexer)*100; 
			 OCCaverageForSGBMErrorMoreThan3[k][i] =(OCCaverageForSGBMErrorMoreThan3[k][i] /lexer)*100; 
			 OCCaverageForSGBMWithinErrorHalf[k][i] = (OCCaverageForSGBMWithinErrorHalf[k][i]/lexer)*100;

			 NOCaverageForSGBMWithinError1[k][i] =(NOCaverageForSGBMWithinError1[k][i]/lexer)*100; 
			 NOCaverageForSGBMWithinError2[k][i] =(NOCaverageForSGBMWithinError2[k][i]/lexer)*100; 
			 NOCaverageForSGBMWithinError3[k][i] =(NOCaverageForSGBMWithinError3[k][i]/lexer)*100; 
			 NOCaverageForSGBMErrorMoreThan3[k][i] =(NOCaverageForSGBMErrorMoreThan3[k][i]/lexer)*100;
			 NOCaverageForSGBMWithinErrorHalf[k][i] = (NOCaverageForSGBMWithinErrorHalf[k][i]/lexer)*100;
			

			}

			
}

		for(int k = 0 ; k<maxCombinations;k++)
		{
					 	 matlabFile<<"yNoc" <<k<<" = [";

						for(int j = 3; j<numOfCensusTypes;j++)
						{
							//	matlabFile<< NOCaverageForCannyEdgeWithinErrorHalf[k][j]<<" "<<NOCaverageForCannyEdgeWithinError1[k][j]<<" "<<NOCaverageForCannyEdgeWithinError2[k][j]<<" "<<NOCaverageForCannyEdgeWithinError3[k][j]<<" "<<NOCaverageForCannyEdgeErrorMoreThan3[k][j] ;	
									matlabFile<< NOCaverageForCannyEdgeWithinErrorHalf[k][j]+NOCaverageForCannyEdgeWithinError1[k][j]+NOCaverageForCannyEdgeWithinError2[k][j]+NOCaverageForCannyEdgeWithinError3[k][j]<<" "<<NOCaverageForCannyEdgeErrorMoreThan3[k][j] ;	

							if(!(j == numOfCensusTypes-1))
					 			matlabFile<<"; ";

						}


					 matlabFile<<"] ;"<<endl;



					 	 	 matlabFile<<"yOcc" <<k<<" = [";

						for(int j =3; j<numOfCensusTypes;j++)
						{
							//	matlabFile<< OCCaverageForCannyEdgeWithinErrorHalf[k][j]<<" "<< OCCaverageForCannyEdgeWithinError1[k][j]<<" "<<OCCaverageForCannyEdgeWithinError2[k][j]<<" "<<OCCaverageForCannyEdgeWithinError3[k][j]<<" "<<OCCaverageForCannyEdgeErrorMoreThan3[k][j] ;	
								matlabFile<< OCCaverageForCannyEdgeWithinErrorHalf[k][j]+ OCCaverageForCannyEdgeWithinError1[k][j]+OCCaverageForCannyEdgeWithinError2[k][j]+OCCaverageForCannyEdgeWithinError3[k][j]<<" "<<OCCaverageForCannyEdgeErrorMoreThan3[k][j] ;	

							if(!(j == numOfCensusTypes-1))
					 			matlabFile<<"; ";

						}


					 matlabFile<<"] ;"<<endl;



				 	 matlabFile<<"sgbmNoc" <<k<<" = [";

					for(int j =3; j<numOfCensusTypes;j++)
					{
						//matlabFile<< NOCaverageForSGBMWithinErrorHalf[k][j]<<" "<< NOCaverageForSGBMWithinError1[k][j]<<" "<<NOCaverageForSGBMWithinError2[k][j]<<" "<<NOCaverageForSGBMWithinError3[k][j]<<" "<<NOCaverageForSGBMErrorMoreThan3[k][j];
						matlabFile<< NOCaverageForSGBMWithinErrorHalf[k][j]+ NOCaverageForSGBMWithinError1[k][j]+NOCaverageForSGBMWithinError2[k][j]+NOCaverageForSGBMWithinError3[k][j]<<" "<<NOCaverageForSGBMErrorMoreThan3[k][j];

							if(!(j == numOfCensusTypes-1))
					 			matlabFile<<"; ";
					}
								 matlabFile<<"] ;"<<endl;



					 matlabFile<<"sgbmOcc" <<k<<" = [";

					for(int j =3; j<numOfCensusTypes;j++)
					{
						//matlabFile<<OCCaverageForSGBMWithinErrorHalf[k][j]<<" "<< OCCaverageForSGBMWithinError1[k][j]<<" "<< OCCaverageForSGBMWithinError2[k][j]<<" "<< OCCaverageForSGBMWithinError3[k][j]<<" "<<OCCaverageForSGBMErrorMoreThan3[k][j];
						matlabFile<<OCCaverageForSGBMWithinErrorHalf[k][j]+OCCaverageForSGBMWithinError1[k][j]+OCCaverageForSGBMWithinError2[k][j]+OCCaverageForSGBMWithinError3[k][j]<<" "<<OCCaverageForSGBMErrorMoreThan3[k][j];

							if(!(j == numOfCensusTypes-1))
					 			matlabFile<<"; ";
					}
								 matlabFile<<"] ;"<<endl;






				for(int j = 3; j<numOfCensusTypes;j++)
						{
								matlabFile<< NOCaverageForCannyEdgeWithinErrorHalf[k][j]<<" "<<NOCaverageForCannyEdgeWithinError1[k][j]<<" "<<NOCaverageForCannyEdgeWithinError2[k][j]<<" "<<NOCaverageForCannyEdgeWithinError3[k][j]<<" "<<NOCaverageForCannyEdgeErrorMoreThan3[k][j] ;	
							//		matlabFile<< NOCaverageForCannyEdgeWithinErrorHalf[k][j]+NOCaverageForCannyEdgeWithinError1[k][j]+NOCaverageForCannyEdgeWithinError2[k][j]+NOCaverageForCannyEdgeWithinError3[k][j]<<" "<<NOCaverageForCannyEdgeErrorMoreThan3[k][j] ;	

							if(!(j == numOfCensusTypes-1))
					 			matlabFile<<"; ";

						}


					 matlabFile<<"] ;"<<endl;



					 	 	 matlabFile<<"yOcc" <<k<<" = [";

						for(int j =3; j<numOfCensusTypes;j++)
						{
								matlabFile<< OCCaverageForCannyEdgeWithinErrorHalf[k][j]<<" "<< OCCaverageForCannyEdgeWithinError1[k][j]<<" "<<OCCaverageForCannyEdgeWithinError2[k][j]<<" "<<OCCaverageForCannyEdgeWithinError3[k][j]<<" "<<OCCaverageForCannyEdgeErrorMoreThan3[k][j] ;	
							//	matlabFile<< OCCaverageForCannyEdgeWithinErrorHalf[k][j]+ OCCaverageForCannyEdgeWithinError1[k][j]+OCCaverageForCannyEdgeWithinError2[k][j]+OCCaverageForCannyEdgeWithinError3[k][j]<<" "<<OCCaverageForCannyEdgeErrorMoreThan3[k][j] ;	

							if(!(j == numOfCensusTypes-1))
					 			matlabFile<<"; ";

						}


					 matlabFile<<"] ;"<<endl;



				 	 matlabFile<<"sgbmNoc" <<k<<" = [";

					for(int j =3; j<numOfCensusTypes;j++)
					{
						matlabFile<< NOCaverageForSGBMWithinErrorHalf[k][j]<<" "<< NOCaverageForSGBMWithinError1[k][j]<<" "<<NOCaverageForSGBMWithinError2[k][j]<<" "<<NOCaverageForSGBMWithinError3[k][j]<<" "<<NOCaverageForSGBMErrorMoreThan3[k][j];
						//matlabFile<< NOCaverageForSGBMWithinErrorHalf[k][j]+ NOCaverageForSGBMWithinError1[k][j]+NOCaverageForSGBMWithinError2[k][j]+NOCaverageForSGBMWithinError3[k][j]<<" "<<NOCaverageForSGBMErrorMoreThan3[k][j];

							if(!(j == numOfCensusTypes-1))
					 			matlabFile<<"; ";
					}
								 matlabFile<<"] ;"<<endl;



					 matlabFile<<"sgbmOcc" <<k<<" = [";

					for(int j =3; j<numOfCensusTypes;j++)
					{
						matlabFile<<OCCaverageForSGBMWithinErrorHalf[k][j]<<" "<< OCCaverageForSGBMWithinError1[k][j]<<" "<< OCCaverageForSGBMWithinError2[k][j]<<" "<< OCCaverageForSGBMWithinError3[k][j]<<" "<<OCCaverageForSGBMErrorMoreThan3[k][j];
						//matlabFile<<OCCaverageForSGBMWithinErrorHalf[k][j]+OCCaverageForSGBMWithinError1[k][j]+OCCaverageForSGBMWithinError2[k][j]+OCCaverageForSGBMWithinError3[k][j]<<" "<<OCCaverageForSGBMErrorMoreThan3[k][j];

							if(!(j == numOfCensusTypes-1))
					 			matlabFile<<"; ";
					}
								 matlabFile<<"] ;"<<endl;



					// cout<<"Line 501"<<endl;
				////1) - 1 edge 
							// 2) - 2 edge
							// 3) - 4 edge
							// 4) - 8 edge
							// 5) - 12 edge
							// 6) - 16 edge
							// 7) - 16 point alternate structure
							// 8) - 16 point diamond structure
							// 9) - 12 point structure
							// 111)- 8 point normal alternate structure
							// 11)- 4 point normal diamond structure
							// 12)- 2 point neighbourhood structure
							// 13)- 1 point neighbourhood structure

					matlabFile<<"\nx = [1:1:13] ; "<<endl;

					for(int j =0 ; j <1;j++)
					{
						matlabFile<<" figure() ; \n subplot(2,1,1); \n bar(x,yOcc" <<k<<");\n ";
						//matlabFile<<"for i = 1:numel(yNoc"<<j+1<<") \n text(yNoc"<<j+1<<"(i) - 0.2, x(i) + 0.4, ['', num2str(x(i))], 'VerticalAlignment', 'top', 'FontSize', 8) \n end"<<endl;
						matlabFile<<"grid on ; \ntitle('CannyEdgeOcc" <<k<<"');"<<endl;
					//	matlabFile<<"\nax=gca; \n ax.XTickLabel = {'withinError1', 'withinError2','withinError3','errorMoreThan3'} "<<endl;
					
						matlabFile<<"\nax=gca; \n ax.XTickLabel = {'1Edge', '2Edge','4Edge','8Edge','12Edge', '16Edge','16PointAl','16PointDia','12Point', '8PointAl','4PointDia','2point','1point'} "<<endl;
						matlabFile<<"ax.XTickLabelRotation=45;"<<endl;
						matlabFile<<"subplot(2,1,2); \n bar(x,sgbmOcc" <<k<<");\n " ; 
						//matlabFile<<"for i = 1:numel(sgbmNoc"<<j+1<<") \n text(sgbmNoc"<<j+1<<"(i) - 0.2, x(i) + 0.4, ['', num2str(x(i))], 'VerticalAlignment', 'top', 'FontSize', 8) \nend"<<endl;		
						matlabFile<<"grid on ; \ntitle('SGBMOcc" <<k<<"');"<<endl;
								matlabFile<<"\nax=gca; \n ax.XTickLabel = {'1Edge', '2Edge','4Edge','8Edge','12Edge', '16Edge','16PointAl','16PointDia','12Point', '8PointAl','4PointDia','2point','1point'} "<<endl;
						matlabFile<<"ax.XTickLabelRotation=45;"<<endl;
						//cout<<"Count j= "<<j<<endl;
						// matlabFile<<"title('Comparison with NOC disparity Maps CensusType "<<censusWindowName.at(j)<<"');"<<endl;
						// matlabFile<<"ax=gca; \n ax.XTickLabel = {'withinError1', 'withinError2','withinError3','errorMoreThan3'} "<<endl;
					}
		}
	
}	




void nonoccSgbmGTCompare(vector<SparseMatch> correspondences, int lexer, string fileType,string filename, Mat SGBMDisp,int censusWinType,int combination)
{
	string fileDirectory = "/home/piyush/Documents/NTU Internship/stereo matching/SparseSetero Code/Baselines/training/disp_noc/";
	Mat GT = imread("disp2.png", -1);

	int totalNumberOfMatchedFeatures = GT.rows*GT.cols;
	int withinError1= 0 ;
	int withinError2= 0 ; 
	int withinError3=0;
	int errorMoreThan3=0;
	int invalidPoint=0;
		int withinErrorHalf=0 ; 

for(int i=0;i<GT.rows;i++)
	for(int j=0;j<(int)GT.cols;j++)
	{
		if((float)(GT.at<uchar>(i,j) ==0))
			invalidPoint++;
		else
		{
		
			float x = abs((float)SGBMDisp.at<short>(i,j)/16.0f- (float)((int)(GT.at<uchar>(i,j)))*(2*(maxDisp/255)));
			if(x<=0.5)
				withinErrorHalf++;
			else if(x <=1 )
				withinError1++;
			else if(x<=2 )
				withinError2++ ; 
			else if(x<=3 )
				withinError3++;
			else if(x>3)
				errorMoreThan3++;

		}
	}
	 NOCSGBMResultwithinError1[combination][censusWinType][lexer] = withinError1;
 	 NOCSGBMResultwithinError2[combination][censusWinType][lexer] = withinError2;
	 NOCSGBMResultwithinError3[combination][censusWinType][lexer] = withinError3;
	 NOCSGBMResulterrorMoreThan3[combination][censusWinType][lexer] = errorMoreThan3;
	 NOCSGBMResultNumberOfFeatures[combination][censusWinType][lexer] = totalNumberOfMatchedFeatures;
 	 NOCSGBMResultinvalidPoint[combination][censusWinType][lexer] = invalidPoint;
 	 NOCSGBMResultwithinErrorHalf[combination][censusWinType][lexer] = withinErrorHalf;
 }



void occSgbmGTCompare(vector<SparseMatch> correspondences, int lexer, string fileType,string filename, Mat SGBMDisp,int censusWinType,int combination)
{
	string fileDirectory = "/home/piyush/Documents/NTU Internship/stereo matching/SparseSetero Code/Baselines/training/disp_occ/";
	Mat GT = imread("disp2.png", -1);

	int totalNumberOfMatchedFeatures = GT.rows*GT.cols ;
	int withinError1= 0 ;
	int withinError2= 0 ; 
	int withinError3=0;
	int errorMoreThan3=0;
	int invalidPoint=0;
	int withinErrorHalf=0 ; 

for(int i=0;i<GT.rows;i++)
	for(int j=0;j<(int)GT.cols;j++)
	{
		if((int)(GT.at<uchar>(i,j))==0 )
			invalidPoint++;
		else
		{
			float x = abs((SGBMDisp.at<short>(i,j)/16.0f)- (float)((int)(GT.at<uchar>(i,j)))*(2*(maxDisp/255)));
			if(x<=0.5)
				withinErrorHalf++;
			else if(x <=1)
				withinError1++;
			else if(x<=2 )
				withinError2++ ; 
			else if(x<=3 )
				withinError3++;
			else if(x>3)
				errorMoreThan3++;


		}
	}
	 OCCSGBMResultwithinError1[combination][censusWinType][lexer] = withinError1;

 	 OCCSGBMResultwithinError2[combination][censusWinType][lexer] = withinError2;
	 OCCSGBMResultwithinError3[combination][censusWinType][lexer] = withinError3;
	 OCCSGBMResulterrorMoreThan3[combination][censusWinType][lexer] = errorMoreThan3;
	 OCCSGBMResultNumberOfFeatures[combination][censusWinType][lexer] = totalNumberOfMatchedFeatures;
 	 OCCSGBMResultinvalidPoint[combination][censusWinType][lexer] = invalidPoint;
 	 OCCSGBMResultwithinErrorHalf[combination][censusWinType][lexer] = withinErrorHalf;
 	
 	}



void occSgbmGTCompareSparse(vector<SparseMatch> correspondences, int lexer, string fileType,string filename, Mat SGBMDisp,int censusWinType,int combination)
{
	string fileDirectory = "/home/piyush/Documents/NTU Internship/stereo matching/SparseSetero Code/Baselines/training/disp_occ/";
	Mat GT = imread("disp2.png", -1);
	int totalNumberOfMatchedFeatures = correspondences.size();
	int withinError1= 0 ;
	int withinError2= 0 ; 
	int withinError3=0;
	int errorMoreThan3=0;
	int invalidPoint=0;
	int withinErrorHalf=0 ; 

	for(int i=0;i<(int)correspondences.size();i++)
	{
		if((float)(GT.at<short>(correspondences[i].rectLeft.y,correspondences[i].rectLeft.x) ==0))
			invalidPoint++;
		else
		{
			float x = abs((float)SGBMDisp.at<short>(correspondences[i].rectLeft.y,correspondences[i].rectLeft.x)/16.0f- (float)((int)GT.at<uchar>(correspondences[i].rectLeft.y,correspondences[i].rectLeft.x)*(float)(maxDisp/255.0)));
			
			if(x<=0.5)
				withinErrorHalf++;
			else if(x <=1 )
				withinError1++;
			else if(x<=2 )
				withinError2++ ; 
			else if(x<=3 )
				withinError3++;
			else if(x>3)
				errorMoreThan3++;


		}
	}
	 OCCSGBMResultwithinError1[combination][censusWinType][lexer] = withinError1;
 	 OCCSGBMResultwithinError2[combination][censusWinType][lexer] = withinError2;
	 OCCSGBMResultwithinError3[combination][censusWinType][lexer] = withinError3;
	 OCCSGBMResulterrorMoreThan3[combination][censusWinType][lexer] = errorMoreThan3;
	 OCCSGBMResultNumberOfFeatures[combination][censusWinType][lexer] = totalNumberOfMatchedFeatures;
 	 OCCSGBMResultinvalidPoint[combination][censusWinType][lexer] = invalidPoint;
 	 OCCSGBMResultwithinErrorHalf[combination][censusWinType][lexer] = withinErrorHalf;
 	
 }



void SgbmGTCompare(vector<SparseMatch> correspondences, int lexer, string fileType,string filename, Mat SGBMDisp,int censusWinType,int combination)
{
	if(sparse == 0 )
	{
		nonoccSgbmGTCompare(correspondences,lexer,fileType,filename,SGBMDisp, censusWinType,combination);
		occSgbmGTCompare(correspondences,lexer,fileType,filename,SGBMDisp, censusWinType,combination);
	}
	else if(sparse == 1)
	{
		nonoccSgbmGTCompareSparse(correspondences,lexer,fileType,filename,SGBMDisp, censusWinType,combination);
		occSgbmGTCompareSparse(correspondences,lexer,fileType,filename,SGBMDisp, censusWinType,combination);
	}

}

void nonoccSgbmGTCompareSparse(vector<SparseMatch> correspondences, int lexer, string fileType,string filename, Mat SGBMDisp,int censusWinType,int combination)
{
	string fileDirectory = "/home/piyush/Documents/NTU Internship/stereo matching/SparseSetero Code/Baselines/training/disp_noc/";
	Mat GT = imread("disp2.png", -1);
	int totalNumberOfMatchedFeatures = correspondences.size();
	int withinError1= 0 ;
	int withinError2= 0 ; 
	int withinError3=0;
	int errorMoreThan3=0;
	int invalidPoint=0;
		int withinErrorHalf=0 ; 

	for(int i=0;i<(int)correspondences.size();i++)
	{
		if((float)(GT.at<short>(correspondences[i].rectLeft.y,correspondences[i].rectLeft.x) ==0))
			invalidPoint++;
		else
		{
		
			float x = abs((float)SGBMDisp.at<short>(correspondences[i].rectLeft.y,correspondences[i].rectLeft.x)/16.0f- (float)((int)GT.at<uchar>(correspondences[i].rectLeft.y,correspondences[i].rectLeft.x)*(float)(maxDisp/255.0)));
			
			if(x<=0.5)
				withinErrorHalf++;
			else if(x <=1 )
				withinError1++;
			else if(x<=2 )
				withinError2++ ; 
			else if(x<=3 )
				withinError3++;
			else if(x>3){
				errorMoreThan3++;
			}

		}
	}
	 NOCSGBMResultwithinError1[combination][censusWinType][lexer] = withinError1;
 	 NOCSGBMResultwithinError2[combination][censusWinType][lexer] = withinError2;
	 NOCSGBMResultwithinError3[combination][censusWinType][lexer] = withinError3;
	 NOCSGBMResulterrorMoreThan3[combination][censusWinType][lexer] = errorMoreThan3;
	 NOCSGBMResultNumberOfFeatures[combination][censusWinType][lexer] = totalNumberOfMatchedFeatures;
 	 NOCSGBMResultinvalidPoint[combination][censusWinType][lexer] = invalidPoint;
 	 NOCSGBMResultwithinErrorHalf[combination][censusWinType][lexer] = withinErrorHalf;
 }



// void CallBackFunc(int event, int x, int y, int flags, void* userdata)
// {
//      if  ( event == EVENT_LBUTTONDOWN )
//      {
//           cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
//      }
  
// }