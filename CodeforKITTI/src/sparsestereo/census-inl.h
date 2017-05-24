/*
 * Author: Konstantin Schauwecker
 * Year:   2012
 */

#ifndef SPARSESTEREO_CENSUS_INL_H
#define SPARSESTEREO_CENSUS_INL_H

#include "exception.h"
#include "census.h"

namespace sparsestereo {
	template<typename T>
	void Census::transform5x5(const cv::Mat_<T>& input, cv::Mat_<unsigned int>* output, int edgeNo) {

		int temp=0;

		if(edgeNo ==7 || edgeNo ==8 )
			temp = 3; 
		else 
			temp =2;
		int maxX=input.cols-temp, maxY=input.rows-temp;

		for(int y=temp; y<maxY; y++)
			for(int x=temp; x<maxX; x++) {
				T centoid = input(y,x);
				switch(edgeNo)
				{
					case 1: (*output)(y,x) = ((input(y, x-1) > input(y, x+1)) ) ; 	//1 edge 
							break;

					case 2: (*output)(y,x) =										//2 edge
							 ((input(y-1, x) > input(y+1, x)) <<1) |
							 ((input(y, x-1) > input(y, x+1)) ) ; 


					case 3: (*output)(y,x) =										//4 edge
							((input(y+1, x+1) > input(y-1, x-1)) << 3) |
							((input(y, x-1) > input(y, x+1)) << 2)	|
							((input(y-1, x+1) > input(y+1, x-1)) << 1) |
							((input(y-1, x) > input(y+1, x)) )	;
							break;

					case 4:  (*output)(y,x) =										//8edge
							 ((input(y+2, x-2) > input(y-2, x+2)) <<7) |
 							 ((input(y-2, x-2) > input(y+2, x+2)) <<6) |
 							 ((input(y, x+2) > input(y, x-2)) <<5) |
 							 ((input(y+2, x) > input(y-2, x)) <<4) |
 							 ((input(y+1, x+1) > input(y-1, x-1)) <<3) |
 							 ((input(y-1, x+1) > input(y+1, x-1)) <<2) |
 							 ((input(y-1, x) > input(y+1, x)) <<1) <<1|
 							  ((input(y, x-1) > input(y, x+1)) <<1) ;



					case 5 :   (*output)(y,x) =									//12 edge
								((input(y-2, x-2) > input(y+2, x+2)) << 11) |
								((input(y, x+2) > input(y, x-2)) << 10) |
								((input(y+2, x-2) > input(y-2, x+2)) << 9) |
								((input(y+2, x) > input(y-2, x)) << 8) |
								((input(y+1, x+1) > input(y-1, x-1)) << 7) |
								((input(y+1, x+1) > input(y, x-1)) << 6) |
								((input(y+1, x-1) > input(y, x+1)) << 5) |
								((input(y-1, x-1) > input(y, x+1)) << 4) |
								((input(y-1, x+1) > input(y+1, x-1)) << 3) |
								((input(y-1, x+1) > input(y, x-1)) << 2) |
								((input(y-1, x) > input(y+1, x)) << 1) |
								((input(y, x-1) > input(y, x+1)) ) ;
								break;

					case 6 :    (*output)(y,x) =									//16 edge
								((input(y-2, x-2) > input(y+2, x+2)) << 15) |
								((input(y+1, x+2) > input(y-1, x-2)) << 14) |
								((input(y, x+2) > input(y, x-2)) << 13) |
								((input(y+1, x-2) > input(y-1, x+2)) << 12) |
								((input(y+2, x-2) > input(y-2, x+2)) << 11) |
								((input(y-2, x+1) > input(y+2, x-1)) << 10) |
								((input(y+2, x) > input(y-2, x)) << 9) |
								((input(y+2, x+1) > input(y-2, x-1)) << 8) |
								((input(y-1, x) > input(y+1, x)) << 7) |
								((input(y+1, x+1) > input(y-1, x-1)) << 6) |
								((input(y+1, x+1) > input(y, x-1)) << 5) |
								((input(y-1, x-1) > input(y, x+1)) << 4) |
								((input(y+1, x-1) > input(y, x+1)) << 3) |
								((input(y-1, x) > input(y+1, x)) << 2) |
								((input(y-1, x+1) > input(y+1, x-1)) << 1) |
								((input(y-1, x+1) > input(y, x-1)) ) ;
								break;

					case 7 : (*output)(y,x) =			//16 point normal alternate structure
					
								((centoid > input(y+3, x+1)) << 15) |
								((centoid > input(y+3, x-1)) <<14) |
					//Row 3
								((centoid > input(y+3, x-3)) << 13) |
								((centoid > input(y+1, x-3)) << 12) |
								((centoid > input(y-1, x-3)) << 11) |
								((centoid > input(y-3, x-3)) << 10) |
								((centoid > input(y-3, x-1)) << 9) |
					// Row 4
								((centoid > input(y-3, x+1)) << 8) |
								((centoid > input(y-3, x+3)) << 7) |
								((centoid > input(y-1, x+3)) << 6) |
								((centoid > input(y+1, x+3)) << 5) |
								((centoid > input(y+3, x+3)) << 4) |
								// Row 5
							
								((centoid > input(y+1, x-1)) << 3) |
								((centoid > input(y-1, x-1)) << 2) |
								((centoid > input(y-1, x+1)) << 1) |
								(centoid > input(y+1, x+1));
									break;

					case 8 : (*output)(y,x) =					//16 point diamond structure	
								((centoid > input(y+3, x)) << 15) |
								((centoid > input(y, x-3)) << 14) |
								((centoid > input(y-3, x)) << 13) |
								((centoid > input(y, x+3)) << 12) |
					// Row 4
								((centoid > input(y+1, x+2)) <<11) |
								((centoid > input(y+2, x+1)) <<10) |
								((centoid > input(y+2, x)) << 9) |
								((centoid > input(y+2, x-1)) << 8) |

								((centoid > input(y+1, x-2)) << 7) |
								((centoid > input(y, x-2)) << 6) |
								((centoid > input(y-1, x-2)) << 5) |
								((centoid > input(y-2, x-1)) << 4) |
								// Row 5
							
								((centoid > input(y-2, x)) << 3) |
								((centoid > input(y-2, x+1)) << 2) |
								((centoid > input(y-1, x+2)) << 1) |
								(centoid > input(y, x+2));
								break;



					case 9 : (*output)(y,x) =			//12 point  structure
								((centoid > input(y+2, x+1)) << 11) |
								((centoid > input(y+2, x)) << 10) |
								((centoid > input(y+2, x-1)) << 9) |
					// Row 4
								((centoid > input(y+1, x-2)) << 8) |
								((centoid > input(y, x-2)) << 7) |
								((centoid > input(y-1, x-2)) << 6) |
								((centoid > input(y-2, x-1)) << 5) |
								((centoid > input(y-2, x)) << 4) |
								// Row 5
							
								((centoid > input(y-2, x+1)) << 3) |
								((centoid > input(y-1, x+2)) << 2) |
								((centoid > input(y, x+2)) << 1) |
								(centoid > input(y+1, x+2));
									break;

					
					case 10 : (*output)(y,x) =			//8 point normal alternate structure
								((centoid > input(y+2, x)) << 7) |
								((centoid > input(y+2, x-2)) << 6) |
								((centoid > input(y, x-2)) << 5) |
								((centoid > input(y-2, x-2)) << 4) |
								// Row 5
							
								((centoid > input(y-2, x)) << 3) |
								((centoid > input(y-2, x+2)) << 2) |
								((centoid > input(y, x+2)) << 1) |
								(centoid > input(y+2, x+2));
									break;


					case 11 : (*output)(y,x) =			//4 point normal diamond structure
								((centoid > input(y+2, x)) << 3) |
								((centoid > input(y, x-2)) << 2) |
								// Row 5
								((centoid > input(y-2, x)) << 1) |
								((centoid > input(y, x+2)) );
									break;

					case 12 : (*output)(y,x) =			//2 point neighbourhood structure
								((centoid > input(y, x-2)) << 1) |
								// Row 5
								((centoid > input(y, x+2)) );
								break;


					case 13 : (*output)(y,x) =			//1 point neighbourhood structure
								((centoid > input(y, x-2)) );
								break;
				}
					// Row 1	//
					//((centoid > input(y-2, x-2)) << 24) |
					// ((centoid > input(y-2, x-1)) << 23) |
					// ((centoid > input(y-2, x)) << 22) |
					// ((centoid > input(y-2, x+1)) << 21) |
					// ((centoid > input(y-2, x+2)) << 20) |
					// // Row 2
					// ((centoid > input(y-1, x-2)) << 19) |
					// ((centoid > input(y-1, x-1)) << 18) |
					// ((centoid > input(y-1, x)) << 17) |
					// ((centoid > input(y-1, x+1)) << 16) |
					// ((centoid > input(y-1, x+2)) <<15) |
					// //Row 3
					// ((centoid > input(y, x-2)) << 14) |
					// ((centoid > input(y, x-1)) << 13) |
					// ((centoid > input(y, x)) << 12) |
					// ((centoid > input(y, x+1)) << 11) |
					// ((centoid > input(y, x+2)) << 10) |
					// // Row 4
					// ((centoid > input(y+1, x-2)) << 9) |
					// ((centoid > input(y+1, x-1)) << 8) |
					// ((centoid > input(y+1, x)) << 7) |
					// ((centoid > input(y+1, x+1)) << 6) |
					// ((centoid > input(y+1, x+2)) << 5) |
					// // Row 5
					// ((centoid > input(y+2, x-2)) << 4) |
					// ((centoid > input(y+2, x-1)) << 3) |
					// ((centoid > input(y+2, x)) << 2) |
					// ((centoid > input(y+2, x+1)) << 1) |
					// (centoid > input(y+2, x+2));
			}
	}
	
	// template <>
	// void Census::transform5x5<char>(const cv::Mat_<char>& input, cv::Mat_<unsigned int>* output);
}

#endif
