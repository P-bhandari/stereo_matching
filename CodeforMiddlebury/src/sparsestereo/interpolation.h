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



// float Yc = 10, Yp = 4, gain =1;
// 	for(int i =0 ; i<disparityImage.rows;i++)
// 		for(int j=0;j<disparityImage.cols;j++)
// 		{
// 			//cout<<"Line 340"<<endl;
// 			if(disparityImage.at<float>(i,j)== 0 && indexHash[labels.at<unsigned int>(i,j)] != -1)
// 			{
// 				int tempLoopSize = boundarySuperPixel.at(indexHash[labels.at<unsigned int>(i,j)]).points.size();
// 				//cout<<"Loop size = "<<tempLoopSize<<endl;
// 				float correlationDenum=0;
// 				for(int k=0 ; k<tempLoopSize ;k++)
// 				{

// 					/*Compute Weight for the pixel*/
// 					int y = boundarySuperPixel.at(indexHash[labels.at<unsigned int>(i,j)]).points.at(k).y;
// 					int x =  boundarySuperPixel.at(indexHash[labels.at<unsigned int>(i,j)]).points.at(k).x ;
// 					float colorDiff = pow( pow((short)leftColorLab.at<cv::Vec3b>(i ,j)[0] - (short)leftColorLab.at<cv::Vec3b>(y,x)[0],2)
// 									+ pow((short)leftColorLab.at<cv::Vec3b>(i ,j)[1] - (short)leftColorLab.at<cv::Vec3b>(y,x)[1],2) 
// 									+ pow((short)leftColorLab.at<cv::Vec3b>(i ,j)[2] - (short)leftColorLab.at<cv::Vec3b>(y,x)[2],2),0.5);
					
// 					float spatialDiff = pow(pow(i-y,2)+pow(j-x,2),0.5);
// 					 float weight =  gain*exp(-((colorDiff/Yc)+(spatialDiff/Yp)));
// 					// cout<<"ColorDiff = "<<colorDiff<<endl;
// 					// cout<<"spatialDiff = "<<spatialDiff<<endl;
// 					// float weight =  gain*((8/(colorDiff+1))+(48/spatialDiff));
				
// 					correlationDenum += weight;

// 					disparityImage.at<float>(i,j) +=weight*(float)disparityImage.at<float>(y,x);
// 				}
// 				disparityImage.at<float>(i,j) = disparityImage.at<float>(i,j)/correlationDenum ;
// 				//cout<<"Disparity at ("<<i<<","<<j<<") = "<<disparityImage.at<float>(i,j)<<endl;
// 			}
// 		}



/*cosine interpolation*/
// for(int i =0 ; i<disparityImage.rows;i++)	//y
// 		for(int j=0;j<disparityImage.cols;j++)	//x
// 		{
// 			if(disparityImage.at<float>(i,j)== 0 && indexHash[labels.at<unsigned int>(i,j)] != -1)
// 	 		{
// 	 			int arrayIndex = indexHash[labels.at<unsigned int>(i,j)];
// 	 			int tempLoopSize = boundarySuperPixel.at(arrayIndex).points.size();
	 			
// 	 			int diffLeft = 10000000 ; 
// 	 			int diffRight = 10000000 ; 

// 	 			int leftPointIndex = -1; 
// 	 			int rightPointIndex = -1;
// 	 			for(int k = 0 ; k < yBasedPoints.at(arrayIndex).y.size();k++)
// 	 			{
// 	 				if(abs(yBasedPoints.at(arrayIndex).y.at(k) - i) < diffLeft && (yBasedPoints.at(arrayIndex).p.at(k).x -j) >0)	//right index
// 	 				{
// 	 					rightPointIndex = k ;
// 	 					diffLeft = abs(yBasedPoints.at(arrayIndex).y.at(k) - i) ; 
// 	 				}

// 	 				if(abs(yBasedPoints.at(arrayIndex).y.at(k) - i) < diffRight && (yBasedPoints.at(arrayIndex).p.at(k).x -j) <0)	//left index
// 	 				{
// 	 					leftPointIndex = k ;
// 	 					diffRight = abs(yBasedPoints.at(arrayIndex).y.at(k) - i) ; 
// 	 				}
// 	 			}

// 	 			if(leftPointIndex == -1 || rightPointIndex == -1)
// 	 			{
// 	 					 float Yc = 12, Yp = 5,gain =1;

// 						int tempLoopSize = boundarySuperPixel.at(indexHash[labels.at<unsigned int>(i,j)]).points.size();
// 						float correlationDenum=0;
// 						for(int k=0 ; k<tempLoopSize ;k++)
// 						{
// 							/*Compute Weight for the pixel*/
// 							int y = boundarySuperPixel.at(indexHash[labels.at<unsigned int>(i,j)]).points.at(k).y;
// 							int x =  boundarySuperPixel.at(indexHash[labels.at<unsigned int>(i,j)]).points.at(k).x ;
// 							float colorDiff = pow( pow((short)leftColorLab.at<cv::Vec3b>(i ,j)[0] - (short)leftColorLab.at<cv::Vec3b>(y,x)[0],2)
// 											+ pow((short)leftColorLab.at<cv::Vec3b>(i ,j)[1] - (short)leftColorLab.at<cv::Vec3b>(y,x)[1],2) 
// 											+ pow((short)leftColorLab.at<cv::Vec3b>(i ,j)[2] - (short)leftColorLab.at<cv::Vec3b>(y,x)[2],2),0.5);
							
// 							float spatialDiff = pow(pow(i-y,2)+pow(j-x,2),0.5);
// 							 float weight =  gain*exp(-((colorDiff/Yc)+(spatialDiff/Yp)));

// 							correlationDenum += weight;

// 							disparityImage.at<float>(i,j) +=weight*(float)disparityImage.at<float>(y,x);
// 						}
// 						disparityImage.at<float>(i,j) = disparityImage.at<float>(i,j)/correlationDenum ;
// 						continue;
// 	 			}
// 	 			double mu =(yBasedPoints.at(arrayIndex).p.at(rightPointIndex).x - j)/(yBasedPoints.at(arrayIndex).p.at(rightPointIndex).x - yBasedPoints.at(arrayIndex).p.at(leftPointIndex).x );
// 	 			double mu2 = (1-cos(mu*PI))/2;
// 	 			double disp2 = disparityImage.at<float>(yBasedPoints.at(arrayIndex).p.at(rightPointIndex).y,yBasedPoints.at(arrayIndex).p.at(rightPointIndex).x);
// 	 			double disp1 = disparityImage.at<float>(yBasedPoints.at(arrayIndex).p.at(leftPointIndex).y,yBasedPoints.at(arrayIndex).p.at(leftPointIndex).x);

// 	 			disparityImage.at<float>(i,j) = disp2*(1-mu2) + disp1*mu2;
// 	 		}
// 		}

/* Sorting the points within the */
	// vector <ySortedPoints> yBasedPoints;
	// ySortedPoints a ; 

	// for(int i = 0 ; i< boundarySuperPixel.size(); i++)
	// {
	// 	a.p.clear();
	// 	a.y.clear();
	// 	for(int j = 0; j<boundarySuperPixel.at(i).points.size();j++)
	// 	{
	// 		a.p.push_back(boundarySuperPixel.at(i).points.at(j)); 
	// 		a.y.push_back(boundarySuperPixel.at(i).points.at(j).y);
	// 	}
	// 	yBasedPoints.push_back(a);
	// }