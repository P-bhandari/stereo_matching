	/*Write all the functions in this part of the file*/




	/*Convert it from sparse to dense*/

	/*Interpolation based on weights of color and distance*/
	// float Yc = 12, Yp = 5,gain =1;
	// for(int i =0 ; i<disparityImage.rows;i++)
	// 	for(int j=0;j<disparityImage.cols;j++)
	// 	{
	// 		//cout<<"Line 340"<<endl;
	// 		if(disparityImage.at<float>(i,j)== 0 && indexHash[labels.at<unsigned int>(i,j)] != -1)
	// 		{
	// 			int tempLoopSize = boundarySuperPixel.at(indexHash[labels.at<unsigned int>(i,j)]).points.size();
	// 			//cout<<"Loop size = "<<tempLoopSize<<endl;
	// 			float correlationDenum=0;
	// 			for(int k=0 ; k<tempLoopSize ;k++)
	// 			{
	// 				/*Compute Weight for the pixel*/
	// 				int y = boundarySuperPixel.at(indexHash[labels.at<unsigned int>(i,j)]).points.at(k).y;
	// 				int x =  boundarySuperPixel.at(indexHash[labels.at<unsigned int>(i,j)]).points.at(k).x ;
	// 				float colorDiff = pow( pow((short)leftColorLab.at<cv::Vec3b>(i ,j)[0] - (short)leftColorLab.at<cv::Vec3b>(y,x)[0],2)
	// 								+ pow((short)leftColorLab.at<cv::Vec3b>(i ,j)[1] - (short)leftColorLab.at<cv::Vec3b>(y,x)[1],2) 
	// 								+ pow((short)leftColorLab.at<cv::Vec3b>(i ,j)[2] - (short)leftColorLab.at<cv::Vec3b>(y,x)[2],2),0.5);
					
	// 				float spatialDiff = pow(pow(i-y,2)+pow(j-x,2),0.5);
	// 				 float weight =  gain*exp(-((colorDiff/Yc)+(spatialDiff/Yp)));
	// 				// cout<<"ColorDiff = "<<colorDiff<<endl;
	// 				// cout<<"spatialDiff = "<<spatialDiff<<endl;
	// 			//	float weight =  gain*((8/(colorDiff+1))+(48/spatialDiff));

	// 				correlationDenum += weight;

	// 				disparityImage.at<float>(i,j) +=weight*(float)disparityImage.at<float>(y,x);
	// 			}
	// 			disparityImage.at<float>(i,j) = disparityImage.at<float>(i,j)/correlationDenum ;
	// 			//cout<<"Disparity at ("<<i<<","<<j<<") = "<<disparityImage.at<float>(i,j)<<endl;
	// 		}
	// 	}






/*Testing the location of similar label boundaries*/
	//  Mat drawing = imread("");;
	// for(int i=0;i<correspondences.size();i++)	
	// {	
	// 	temp.label = -1;
	// 	index = labels.at<unsigned int>(correspondences.at(i).imgLeft->pt.y,correspondences.at(i).imgLeft->pt.x);
	// 	Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );

	// 	if(indexHash[index] != -1)
	// 	{
			
	// 			for(int k=0 ; k<boundarySuperPixel.at(indexHash[index]).points.size() ;k++)
	// 			{
	// 				int y = boundarySuperPixel.at(indexHash[index]).points.at(k).y;
	// 				int x =  boundarySuperPixel.at(indexHash[index]).points.at(k).x ;
	// 				//cout<<"Label  = "<<labels.at<unsigned int>(y,x)<<endl;
	// 				cv::circle(drawing,Point(x,y),2,color,1,8,0);
	// 			}
			

	// 	}
		
	// }