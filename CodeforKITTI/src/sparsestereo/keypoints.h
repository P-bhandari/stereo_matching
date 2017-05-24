
#include <opencv2/ximgproc.hpp>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;
using namespace boost::posix_time;

vector<KeyPoint> getKeyPoints(Mat image, Mat& labels, Mat& mask, int lexer,Ptr<SuperpixelSEEDS>& seeds)
{


    
		Mat frame = image.clone(); //has to be a colored image
       // ptime lastTime = microsec_clock::local_time();

		if(image.channels()<3)
			cout<<"Please specify a colored Image"<<endl;
		
		int num_iterations = 4;
    	

        Mat converted;
      
        cvtColor(frame, converted, COLOR_BGR2HSV);
        seeds->iterate(converted, num_iterations);
        Mat result ;
        result = frame;


        //Total Number of superPixels
        int totalSuperPixels=  seeds->getNumberOfSuperpixels();
     //   cout<<"Total Superpixels = "<<totalSuperPixels<<endl;
        //Labelled Image
        seeds->getLabels(labels);
 		

 		/* get the contours for displaying */
        seeds->getLabelContourMask(mask, false);
       // time_duration elapsed = (microsec_clock::local_time() - lastTime);
      //  cout << "SuperPixel and label and mask computation " << elapsed.total_microseconds()/1.0e6 << "s" << endl;
        result.setTo(Scalar(0, 0, 255), mask);
        // Mat edges;
        // Canny( frame,edges, 80, 160,3 );

        // cv::bitwise_or(edges,mask,mask,noArray());
        // imshow("Mask",mask);
        // waitKey();
        vector<KeyPoint> keypoints ;
        // imshow("Results",result);
        // waitKey();
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours( mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0) );
		for(int i=0; i< contours.size();i++)
		{
			 for(int j=0;j<contours[i].size();j++)
			  {
		    		keypoints.push_back(cv::KeyPoint(contours[i].at(j), 1.f));
                  //  cout<<"Label of ("<<i<<","<<j<<")"<<"="<<labels.at<unsigned int>(contours[i].at(j).y, contours[i].at(j).x) <<endl;

			  }
		}
     //   time_duration elapsed2 = (microsec_clock::local_time() - lastTime);

        // cout << "Time for total Stuff including storing of all the keypoints " << elapsed2.total_microseconds()/1.0e6 << "s" << endl;
        // cout<<"Input feature size ="<<keypoints.size()<<endl;
        // vector<int> compression_params;
        // compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
        // compression_params.push_back(9);    
        // stringstream ss;
        // ss << lexer;
        // string fileNumber = ss.str();
        // imwrite("/home/piyush/Documents/NTU Internship/stereo matching/SparseSetero Code/Baselines/training/outputs/200/doubleCheckTrue/SupP"+fileNumber+".png", result, compression_params);
        
        return keypoints;

 }



// vector<KeyPoint> getKeyPoints(Mat image)
// {
//      //cout<<"Line 151"<<endl;

//      Mat inputImage = image.clone(); //has to be 8 bit single channel 
//              vector<KeyPoint> keypoints ;


//      // cout<<"image cols = " <<inputImage.cols;
//      // cout<<"image rows = " <<inputImage.rows;


//      for(int i=0; i<inputImage.rows;i++)
//          for(int j=0; j<inputImage.cols;j++)
//          {
//              keypoints.push_back(cv::KeyPoint(Point(j,i),1.f));
//          }
//  //      cout<<"Keypoints Size = "<<keypoints.size()<<endl;

//           return keypoints;


// }
// vector<KeyPoint> getKeyPoints(Mat image)
// {
//         //cout<<"Line 151"<<endl;

//         Mat inputImage = image.clone(); //has to be 8 bit single channel
//         //  cout<<"Line 154"<<endl;

//         Mat edges;
//         Canny( inputImage,edges, 20, 20*10,3 );
//         cout<<"image rows = "<<image.rows<<endl;
//         cout<<"image cols = "<<image.cols<<endl;

//     //  imshow("edges",edges);
//     //  RNG rng(12345);
//         vector<KeyPoint> keypoints ;

//          vector<vector<Point> > contours;
//           vector<Vec4i> hierarchy;
//           findContours(edges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));
//                 //      cout<<"Line 167"<<endl;

//           for(int i=0; i< contours.size();i++)
//           {
//               for(int j=0;j<contours[i].size();j++)
//               {
//                     keypoints.push_back(cv::KeyPoint(contours[i].at(j), 1.f));
//               }
//           }
//           cout<<"Feature size = "<<keypoints.size()<<endl;
//              return keypoints;


// }