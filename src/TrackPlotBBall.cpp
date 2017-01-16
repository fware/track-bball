#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "DebugHelpers.hpp"
//#include <opencv2/legacy/compat.hpp>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>


//#define IMG_DEBUG
#define SHOT_DEBUG

using namespace std;
using namespace cv;

class PlayerObs {
	public:
		PlayerObs();
	public:
		int		activeValue;
		int		radiusIdx;
		int 	placement;
		Point   position; 
		int 	frameCount;
};

PlayerObs::PlayerObs() : activeValue( 0 ), radiusIdx( 0 ), placement( 0 ), position(0, 0), frameCount( 0 )
{}


void getGray(const Mat& image, Mat& gray);
double DistanceToCamera(double knownWidth, double focalLength, double perWidth);
int getPlayerOffsetX(int frame_count, float player_pos_x, float player_pos_y, int player_hgt);
int getPlayerOffsetY(int frame_count, float player_pos_x, float player_pos_y, int player_hgt);
vector<Point> get_sliding_windows(Mat& image, int winWidth);
Mat drawSemiCircle(Mat& image, int radius, Point center);
//float euclideanDist(Point& p, Point& q);
double euclideanDist(double x1, double y1, double x2, double y2);
double oneDDist(double p1, double p2);
int findIndex_BSearch(const vector< int> &my_numbers, int key);

static void help()
{
 printf("\nUsing various functions in opencv to track a basketball.\n"
"			./TrackPlot {file index number} (choose 1 or 4)\n\n");
}

int main(int argc, const char** argv)
{
	//const string videofileName = argc >= 2 ? argv[1] : "v1.mp4";
	const string videoIdx 							= argc >= 2 ? argv[1] : "1";
	int fileNumber;
	if ( argc > 1 ) {
		fileNumber = atoi( argv[1] );
	}
	else {
		fileNumber = 1;
	}
	stringstream vSS;
	vSS << fileNumber;
    string vIdx 									= vSS.str();
	const string videofileName 						= "/home/fred/Videos/testvideos/v" + vIdx + ".mp4";
    help();
	int frameCount 									= 0;
	int sleepDelay 									= 0;
	const string bballPatternFile 					= "/home/fred/Pictures/OrgTrack_res/bball3_vga.jpg";
	Mat patternImage 								= imread(bballPatternFile);
	const string bballFileName 						= "/home/fred/Pictures/OrgTrack_res/bball-half-court-vga.jpeg";
    VideoCapture cap(videofileName);
    Mat bbsrc 										= imread(bballFileName);	
	int thresh 										= 85;
    int max_thresh 									= 255;
	RNG rng(12345);
	int backboardOffsetX							= 0;
	int backboardOffsetY							= 0;
	int newPlayerWindowSize 						= 50;
	PlayerObs newPlayerWindow[newPlayerWindowSize];  
	float hgtThresh1 								= 0.9;  //0.5;  0.6;
	float hgtThresh2 								= 1.0;  //0.6;  0.7;
	vector <int> hTableRange;
	Point hgtRings[newPlayerWindowSize][1200 /*562*/];
	Mat threshold_output;
	vector<Vec4i> hierarchy;
//    namedWindow("image", WINDOW_NORMAL);
	namedWindow("halfcourt", WINDOW_NORMAL);

	Ptr<BackgroundSubtractor> bg_model;
    bg_model 										= createBackgroundSubtractorMOG2(30, 16.0, false);
    Mat img, fgmask, fgimg;
	Mat grayForRect;
	Rect bballRect;
	vector< vector<Point> > boardContours;	
	Scalar redColor 								= Scalar (0, 20, 180);
	Scalar greenColor 								= Scalar (0, 215, 0);
	Scalar blueColor 								= Scalar (180, 0, 0);
	bool haveBackboard 								= false;
    vector<Rect> bodys;
	String body_cascade_name 						= "/home/fred/Pictures/OrgTrack_res/cascadeconfigs/haarcascade_fullbody.xml";
	CascadeClassifier body_cascade; 
	Mat imgBBallGray;
	Mat firstFrame;
	bool isPatternPresent;
	bool semiCircleReady 							= false;
	Rect offsetBackboard;
	Rect freezeBB;
	Point bodyPosit;
	Point bbCenterPosit;
	int freezeCenterX;
	int freezeCenterY;
	int leftImgBoundary;
	int rightImgBoundary;
	int topImgBoundary;
	int bottomImgBoundary;
	Point   mostRecentPosition; 

	const string OUTNAME = "v4_output_longversion.mp4";	

    if( !cap.isOpened() )
    {
		cout << "can not open video file " << videofileName << endl;
        return -1;
    }


#ifdef IMG_DEBUG
	int l_bound1=260; int r_bound1=305;
	int l_bound2=600; int r_bound2=700;
	int l_bound3=700; int r_bound3=800;
	int l_bound4=1000; int r_bound4=1300;
	int l_bound5=1300; int r_bound5=1400;
	int l_bound6=1400; int r_bound6=1500;
	int l_bound7=1500; int r_bound7=1600;
#endif

	cap >> firstFrame;
	if (firstFrame.empty())
	{
        std::cout << "Cannot retrieve first video capture frame." << std::endl;
        return -1;
	}

    int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form
    Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	Size outS = Size ((int) 2 * S.width, S.height);
	VideoWriter outputVideo; 
	outputVideo.open(OUTNAME, ex, cap.get(CV_CAP_PROP_FPS), outS, true);
	Mat finalImg(S.height, S.width+S.width, CV_8UC3);

	leftImgBoundary 			= firstFrame.cols/4;  
	rightImgBoundary			= firstFrame.cols*3/4;
	topImgBoundary				= firstFrame.rows/4;
	bottomImgBoundary			= firstFrame.rows*3/4;
   
	firstFrame.release();
	
    for(;;)
    {
        cap >> img; 	
		frameCount++;
		sleepDelay = 0;
		
        if( img.empty() )
            break;

		if( !body_cascade.load( body_cascade_name ) )
		{ 
			printf("--(!)Error loading body_cascade_name\n"); return -1; 
		}

		getGray(img,grayForRect);
		blur(grayForRect, grayForRect, Size(3,3));
		equalizeHist(grayForRect, grayForRect);
		threshold(grayForRect,threshold_output,thresh,255, THRESH_BINARY);
		findContours( threshold_output, boardContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

		vector< vector<Point> > contours_poly( boardContours.size() );
		vector<Rect> boundRect( boardContours.size() );
		vector<Point2f> center( boardContours.size() );
		vector<float> rradius( boardContours.size() );

		///*************************Start of main code to detect BackBoard*************************		
		for ( int i = 0; i < boardContours.size(); i++ ) {
			approxPolyDP(Mat(boardContours[i]),contours_poly[i],3,true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
			minEnclosingCircle( (Mat) contours_poly[i], center[i], rradius[i] );
		}
		
		double bb_ratio = 0.0;
		double bb_w = 0.0;
		double bb_h = 0.0;
		double bb_area = 0.0;
		int bb_x, bb_y;
		///*************************End of main code to detect BackBoard*************************

		///*******Start of main code to detect Basketball*************************
        if( fgimg.empty() )
          fgimg.create(img.size(), img.type());

        bg_model->apply(img, fgmask);

        fgimg = Scalar::all(0);
        img.copyTo(fgimg, fgmask);

		Canny(fgmask, fgmask, thresh, thresh*2, 3);
		
		vector<vector<Point> > bballContours;
		vector<Vec4i> hierarchy;
		findContours(fgmask,bballContours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
		
		Mat imgBball = Mat::zeros(fgmask.size(),CV_8UC3);
		for (int i = 0; i < bballContours.size(); i++ )
		{
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
			drawContours(imgBball,bballContours,i,color,2,8,hierarchy,0,Point());
		}


		//------------Track the basketball!!!!---------------
        vector<Vec3f> basketballTracker;
		getGray(imgBball, imgBBallGray);
		float canny1 = 100;
        float canny2 = 14; //16;
		double minDist = imgBBallGray.rows/8; //4;
		HoughCircles(imgBBallGray, basketballTracker, CV_HOUGH_GRADIENT, 1, minDist, canny1, canny2, 1, 9 );

		if (basketballTracker.size() > 0)
		{
			for (size_t i = 0; i < basketballTracker.size(); i++)
			{		
				Point bballCenter(cvRound(basketballTracker[i][0]), cvRound(basketballTracker[i][1]));
				double bballRadius = (double) cvRound(basketballTracker[i][2]);
				double bballDiameter = (double)(2*bballRadius);

				int bballXtl = (int)(basketballTracker[i][0]-bballRadius);
				int bballYtl = (int)(basketballTracker[i][1]-bballRadius);
				bballRect = Rect(bballXtl, bballYtl, bballDiameter, bballDiameter);

				//-------We chose what we think is the basketball and put it in basketballChoice!!!---------
				Rect basketballChoice = bballRect;
				
				for (size_t j = 0; j < boardContours.size(); j++) 
				{
					bb_w = (double) boundRect[j].size().width;
					bb_h = (double) boundRect[j].size().height;
					bb_ratio = bb_w/bb_h;

                    //-----------Find the Backboard!!!-----------------
					if((boundRect[j].area() > 700)
						&& (boundRect[j].area() < 900)
						&& (bb_ratio > 1.50) 
						&& (bb_ratio < 2.00)) {

						if (fileNumber <= 3) 
						{
							if (boundRect[j].tl().x < img.size().width/2) 
							{
							    if (!haveBackboard) 
								{
									backboardOffsetX = -boundRect[j].tl().x + img.size().width/2 - 13;
									backboardOffsetY = -boundRect[j].tl().y + 30;
									offsetBackboard = Rect(boundRect[j].tl().x+backboardOffsetX, 
													boundRect[j].tl().y+backboardOffsetY, 
													boundRect[j].size().width,
													boundRect[j].size().height);
									freezeBB = boundRect[j];
							    }
								haveBackboard = true;
								freezeCenterX = (freezeBB.tl().x+(freezeBB.width/2));
								freezeCenterY = (freezeBB.tl().y+(freezeBB.height/2));
							}
						}
						else if (fileNumber == 4) 
						{
							if (boundRect[j].tl().x > img.size().width/2) 
							{
							    if (!haveBackboard) 
								{
									//----------We chose our background and put it in freezeBB!!!--------------
									freezeBB = boundRect[j];

									//----------Compute the offset for backboard on shot chart!!!---------------
									backboardOffsetX = -boundRect[j].tl().x + img.size().width/2 - 13;
									backboardOffsetY = -boundRect[j].tl().y + 30;
									offsetBackboard = Rect(boundRect[j].tl().x+backboardOffsetX, 
													boundRect[j].tl().y+backboardOffsetY, 
													boundRect[j].size().width,
													boundRect[j].size().height);
							    }
								haveBackboard = true;
								freezeCenterX = (freezeBB.tl().x+(freezeBB.width/2));
								freezeCenterY = (freezeBB.tl().y+(freezeBB.height/2));
							}
						}
					}
					//**** End of selection of backboard rectangle
				}

				if ((bballRect.x > leftImgBoundary /*img.cols/4*/) 
								&& (bballRect.x < rightImgBoundary /*img.cols * 3 / 4*/)
								&& (bballRect.y > topImgBoundary /*img.rows/4*/)
								&& (bballRect.y < bottomImgBoundary /*img.rows * 3 / 4*/)) {
					sleepDelay = 0;

					//The basketball on video frames.
					Scalar rngColor = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
					Rect objIntersect = freezeBB & bballRect;

					//rectangle(img, basketballChoice.tl(), basketballChoice.br(), Scalar(0,180,255), 2, 8, 0 );
					//rectangle(img, freezeBB.tl(), freezeBB.br(), Scalar(180,50,0), 2, 8, 0 );


					//---Start of the process of identifying a shot at the basket!!!------------
					if (objIntersect.area() > 0) {
						
						//---Start of using player position on halfcourt image to draw shot location-----
						int activeCount = 0;
						Point playerPositAvg = Point(0, 0);
						int playerHeightIndex = 0;
						int playerPlacement = 0;
						
						for (int k=30; k < newPlayerWindowSize; k++) 
						{
							activeCount += newPlayerWindow[k].activeValue;
							playerPositAvg += newPlayerWindow[k].position;
							playerHeightIndex += newPlayerWindow[k].radiusIdx;
							playerPlacement += newPlayerWindow[k].placement;
						}

						if (activeCount > 0) 
						{
							playerPositAvg /= activeCount;
							playerHeightIndex /= activeCount;
							playerPlacement /= activeCount;
						}

						if (playerPositAvg.x == 0 || playerPositAvg.y == 0) 
						{
							playerPositAvg = newPlayerWindow[newPlayerWindowSize-1].position;
							playerHeightIndex = newPlayerWindow[newPlayerWindowSize-1].radiusIdx;
							playerPlacement = newPlayerWindow[newPlayerWindowSize-1].placement;
							//cout << "playerPositAvg[newPlayerWindowSize-1]=" << playerPositAvg
							//	 << "   playerHeightIndex[newPlayerWindowSize-1]=" << playerHeightIndex
							//	 << "   playerPlacement[newPlayerWindowSize-1]=" << playerPlacement
							//	 << endl;
							cout << "newPlayerWindow[0].frameCount=" << newPlayerWindow[0].frameCount
							     << "newPlayerWindow[0].position=" << newPlayerWindow[0].position
								 << "   newPlayerWindow[0].radiusIdx=" << newPlayerWindow[0].radiusIdx
								 << "   newPlayerWindow[0].placement=" << newPlayerWindow[0].placement
								 << endl;
						}

						//circle(img, bballCenter, 3, Scalar(0,255,255), -1);
						if (frameCount > 220) 
						{
							//if ((frameCount - newPlayerWindow[0].frameCount) < 75) 
							//{
								//if ((playerPositAvg.x == 0) || (playerPositAvg.y == 0))
								if ((newPlayerWindow[0].position.x == 0) || (newPlayerWindow[0].position.y == 0))
								{
									cout << "newPlayerWindow[0].frameCount=" << newPlayerWindow[0].frameCount
										 << "	 freezeCenter="		   		<< Point(freezeCenterX,freezeCenterY)
										 << "	 Chosen  newPlayerWindow[0].position="	<<	newPlayerWindow[0].position
										 << "	 hgtRings["					<< newPlayerWindow[0].radiusIdx
										 << "][" 							<< newPlayerWindow[0].placement
										 << "]=" 							<< hgtRings[newPlayerWindow[0].radiusIdx][newPlayerWindow[0].placement]
										 << endl;
									circle(bbsrc, hgtRings[newPlayerWindow[0].radiusIdx][newPlayerWindow[0].placement], 1, Scalar(0, 165, 255), 3); 
								}
								else
								{
									//cout << frameCount << ":"
									//	 << "	 freezeCenter="		   		<< Point(freezeCenterX,freezeCenterY)
									//	 << "	 Chosen  playerPositAvg="	<<	playerPositAvg
									//	 << "	 hgtRings["					<< playerHeightIndex
									//	 << "]["							<< playerPlacement
									//	 << "]="							<< hgtRings[playerHeightIndex][playerPlacement]
									//	 << endl;
									cout << frameCount << ":"
										 << "	 freezeCenter="		   		<< Point(freezeCenterX,freezeCenterY)
										 << "	 Chosen  newPlayerWindow[0].position="	<< newPlayerWindow[0].position
										 << "	 hgtRings["					<< newPlayerWindow[0].radiusIdx
										 << "]["							<< newPlayerWindow[0].placement
										 << "]="							<< hgtRings[newPlayerWindow[0].radiusIdx][newPlayerWindow[0].placement]
										 << endl;
									circle(bbsrc, hgtRings[newPlayerWindow[0].radiusIdx][newPlayerWindow[0].placement], 1, Scalar(0, 165, 255), 3);
								}
							//}
						}

						Point semiCircleCenterPt( (offsetBackboard.tl().x+offsetBackboard.width/2) , (offsetBackboard.tl().y + offsetBackboard.height/2) );
						bbCenterPosit = semiCircleCenterPt;
						
						if (!semiCircleReady) {
							int bCounter = 0;
							for (int radius=40; radius < 280; radius+= 20)   //Radius for distFromBB
							{
								hTableRange.push_back(radius);
								
								int temp1, temp2, temp3;
								int yval;
								for (int x=bbCenterPosit.x-radius; x<=bbCenterPosit.x+radius; x++) 
								{
									temp1 = radius * radius;
									temp2 = (x - bbCenterPosit.x) * (x - bbCenterPosit.x);
									temp3 = temp1 - temp2;
									yval = sqrt(temp3);
									yval += bbCenterPosit.y;
									Point ptTemp = Point(x, yval);
									hgtRings[bCounter][x] = ptTemp;
									//cout << "hgtRings[" << bCounter << "][" << x << "]=" << ptTemp << endl;
									//circle(bbsrc, ptTemp, 1, Scalar(0,255,0), -1);			
								}
								
								bCounter++;
							}
							semiCircleReady = true;
						}
						//circle(bbsrc, semiCircleCenterPt, 1, Scalar(180, 50, 230), 3);
						//rectangle(bbsrc, offsetBackboard.tl(), offsetBackboard.br(), blueColor, 2, 8, 0 );
					}
					//---Start of using player position on halfcourt image to draw shot location-----
					//---End of the process of identifying a shot at the basket!!!------------
					
				}				
			}
			///*******End of code to detect & select Basketball*************************
    	}

	    //-- detect body 
	    body_cascade.detectMultiScale(grayForRect, bodys, 1.1, 2, 18|9, Size(3,7));

		stringstream ss;
		ss << frameCount;
		stringstream bodyheight;
		stringstream bodycenterX;
		stringstream bodycenterY;
		string posit_on_video;
		string bb_initial_inputs;
		string posit_on_chart;
		stringstream playerXStr;
		stringstream playerYStr;

/*
		if (bodys.size() == 0) 
		{
			//Do nothing
			for (int k = newPlayerWindowSize; k > 1; k--) 
			{
				newPlayerWindow[k-1].activeValue = newPlayerWindow[k-2].activeValue;
				newPlayerWindow[k-1].radiusIdx = newPlayerWindow[k-2].radiusIdx;
				newPlayerWindow[k-1].placement = newPlayerWindow[k-2].placement;
				newPlayerWindow[k-1].position = newPlayerWindow[k-2].position;
				newPlayerWindow[k-1].frameCount = newPlayerWindow[k-2].frameCount;
			}
			newPlayerWindow[0].activeValue = 0;
			newPlayerWindow[0].radiusIdx = 0;
			newPlayerWindow[0].placement = 0;
			newPlayerWindow[0].position = Point(0,0);
			newPlayerWindow[0].frameCount = frameCount;
		}
		else 
		{
*/		
		    for( int j = 0; j < bodys.size(); j++ ) 
	        {
	        	//-----------Identifying player height and position!!--------------
				Point bodyCenter( bodys[j].x + bodys[j].width*0.5, bodys[j].y + bodys[j].height*0.5 ); 
				
				//--- Start of adjusting player position on image of half court!!!-----
				// NOTE:  If player height is greater that 135 just skip the calculations and adjustments for this particular frame.  It is too out of scope.
				//if (bodys[j].height <= 135) 
				//{
					

					//Sliding window for finding the average position of the player.
					for (int k = newPlayerWindowSize; k > 1; k--) {
						newPlayerWindow[k-1].activeValue = newPlayerWindow[k-2].activeValue;
						newPlayerWindow[k-1].radiusIdx = newPlayerWindow[k-2].radiusIdx;
						newPlayerWindow[k-1].placement = newPlayerWindow[k-2].placement;
						newPlayerWindow[k-1].position = newPlayerWindow[k-2].position;
						newPlayerWindow[k-1].frameCount = newPlayerWindow[k-2].frameCount;
					}
					newPlayerWindow[0].frameCount = frameCount;
					newPlayerWindow[0].activeValue = 1;
					newPlayerWindow[0].position = bodyCenter; //playerNewPosit;
					

					double distFromBB = euclideanDist((double) freezeCenterX,(double) freezeCenterY,(double) bodyCenter.x, (double) bodyCenter.y);
					double xDistFromBB = oneDDist(freezeCenterX, bodyCenter.x);
					double yDistFromBB = oneDDist(freezeCenterY, bodyCenter.y);

#ifdef SHOT_DEBUG 
					cout << "   frameCount=" 		<< frameCount
						 << "   freezeCenter="	    << Point(freezeCenterX, freezeCenterY)
						 << "   bodyCenter=" 		<< bodyCenter
					 	 << "   bodys[j].height="   << bodys[j].height 
						 << "   distFromBB=" 		<< distFromBB
						 << "   xDistFromBB="		<< xDistFromBB
						 << "   yDistFromBB="		<< yDistFromBB
						 << endl;
#endif

					if (distFromBB > 135) 
					{
						newPlayerWindow[0].radiusIdx = hTableRange.size() * 0.99;

						distFromBB += 120;

						cout << "frameCount="										<< frameCount
							 << "    bbCenterPosit.x="								<< bbCenterPosit.x
							 << "    hTableRange[newPlayerWindow[0].radiusIdx]="	<< hTableRange[newPlayerWindow[0].radiusIdx]
							 << endl;

						int tempPlacement = (bbCenterPosit.x + hTableRange[newPlayerWindow[0].radiusIdx])
										- (bbCenterPosit.x - hTableRange[newPlayerWindow[0].radiusIdx]);
						
						if (bodyCenter.x > freezeCenterX) tempPlacement -= 1;
						else tempPlacement = 0;
						tempPlacement += (bbCenterPosit.x - hTableRange[newPlayerWindow[0].radiusIdx]);

						newPlayerWindow[0].placement = tempPlacement;
					}
					else if (distFromBB < 30) 
					{
						int tempPlacement;
						if (bodyCenter.x < freezeCenterX) tempPlacement = 0;
					    else tempPlacement = (bbCenterPosit.x + hTableRange[newPlayerWindow[0].radiusIdx])
										- (bbCenterPosit.x - hTableRange[newPlayerWindow[0].radiusIdx]) - 1;
						
						newPlayerWindow[0].placement = tempPlacement;
						newPlayerWindow[0].radiusIdx = hTableRange.size() * 0.01;
					}
					else 
					{
					    if (bodys[j].height < 170)    //NOTE:  If not true, then we have inaccurate calculation of body height from detectMultiscale method.  Do not estimate a player position for it. 
						{
							newPlayerWindow[0].radiusIdx = findIndex_BSearch(hTableRange, distFromBB /*bodys[j].height*/ /*hgtAdjustment*/);
							newPlayerWindow[0].radiusIdx += 5;
							if ((xDistFromBB < 51) && (yDistFromBB < 70)) newPlayerWindow[0].radiusIdx = 0;
							cout << "frameCount="	<< frameCount
								 << "	 radiusIdx=" << newPlayerWindow[0].radiusIdx
								 << endl;
							
							double percentPlacement = (double) (bodyCenter.x - leftImgBoundary) / (rightImgBoundary - leftImgBoundary);
							int leftRingBound		= bbCenterPosit.x - hTableRange[newPlayerWindow[0].radiusIdx];
							int rightRingBound		= bbCenterPosit.x + hTableRange[newPlayerWindow[0].radiusIdx];
							int chartPlacementTemp	= (rightRingBound - leftRingBound) * percentPlacement;
							int chartPlacement		= leftRingBound + chartPlacementTemp;

							newPlayerWindow[0].placement = chartPlacement;
						}
					}
					//newPlayerWindow[0].frameCount = frameCount;    NOTE:  This wrong place to save the correct frame number.  Should always record it in conjunction with position.
					
					//--- End of adjusting player position on image of half court!!!-----

#ifdef IMG_DEBUG				
					posit_on_video = "video v" + vIdx + "  frame" + ss.str() + "  body height:" + bodyheight.str() + " body center(" + bodycenterX.str() + "," + bodycenterY.str() + ")";
					posit_on_chart = "chart v" + vIdx + "  frame" + ss.str() + "  playerPosit(" + playerXStr.str() + "," + playerYStr.str() + ")  for body height=" + bodyheight.str();
					if (frameCount > l_bound1 && frameCount < r_bound1) {
						printf("%d:playerBodyCenter(%f, %f)   playerNewPosit=(%d, %d)\n", frameCount, playerBodyCenterX, playerBodyCenterY, playerNewPosit.x, playerNewPosit.y);
						circle(bbsrc, playerNewPosit, 1, redColor, -1);
						showAndSave( posit_on_chart, bbsrc);
					}
					else if (frameCount > l_bound2 && frameCount < r_bound2) {
						printf("%d:playerNewPosit=(%d, %d)\n", frameCount, playerNewPosit.x, playerNewPosit.y);
						circle(bbsrc, playerNewPosit, 1, greenColor, -1);
						showAndSave( posit_on_chart, bbsrc);
					}
					else if (frameCount > l_bound3 && frameCount < r_bound3) {
						printf("%d:playerNewPosit=(%d, %d)\n", frameCount, playerNewPosit.x, playerNewPosit.y);
						circle(bbsrc, playerNewPosit, 1, blueColor, -1);
						showAndSave( posit_on_chart, bbsrc);
					}
					else if (frameCount > l_bound4 && frameCount < r_bound4) {
						printf("%d:playerNewPosit=(%d, %d)\n", frameCount, playerNewPosit.x, playerNewPosit.y);
						circle(bbsrc, playerNewPosit, 1, blueColor, -1);
						showAndSave( posit_on_chart, bbsrc);
					}
					else if (frameCount > l_bound5 && frameCount < r_bound5) {
						printf("%d:playerNewPosit=(%d, %d)\n", frameCount, playerNewPosit.x, playerNewPosit.y);
						circle(bbsrc, playerNewPosit, 1, blueColor, -1);
						showAndSave( posit_on_chart, bbsrc);
					}
					else if (frameCount > l_bound6 && frameCount < r_bound6) {
						printf("%d:playerNewPosit=(%d, %d)\n", frameCount, playerNewPosit.x, playerNewPosit.y);
						circle(bbsrc, playerNewPosit, 1, blueColor, -1);
						showAndSave( posit_on_chart, bbsrc);
					}		
#endif

					//ellipse( img, bodyCenter, Size( bodys[j].width*0.5, bodys[j].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
				//}
	        } 
//		}// end else
		

		//Create string of frame counter to display on video window.
		string str = "frame" + ss.str();		
		putText(img, str, Point(100, 100), FONT_HERSHEY_PLAIN, 2 , greenColor, 2);

		Mat left(finalImg, Rect(0, 0, img.cols, img.rows));
		img.copyTo(left);
		Mat right(finalImg, Rect(bbsrc.cols, 0, bbsrc.cols, bbsrc.rows));
		bbsrc.copyTo(right);		

		imshow("halfcourt", finalImg); //bbsrc);
        //imshow("image", img);

#ifdef IMG_DEBUG
		if (frameCount > l_bound1 && frameCount < r_bound1) 
			showAndSave( posit_on_video, img);
		else if (frameCount > l_bound2 && frameCount < r_bound2) 
			showAndSave( posit_on_video, img);
		else if (frameCount > l_bound3 && frameCount < r_bound3) 
			showAndSave( posit_on_video, img);
		else if (frameCount > l_bound4 && frameCount < r_bound4) 
			showAndSave( posit_on_video, img);
		else if (frameCount > l_bound5 && frameCount < r_bound5) 
			showAndSave( posit_on_video, img);
		else if (frameCount > l_bound6 && frameCount < r_bound6) 
			showAndSave( posit_on_video, img);
#endif
			
		sleep(sleepDelay);

        char k = (char)waitKey(30);
        if( k == 27 ) break;

		outputVideo << finalImg;

		//if (frameCount == 2000) break;    //NOTE:  Break out of loop early.
    }

    return 0;
}


void getGray(const Mat& image, Mat& gray)
{
    if (image.channels()  == 3)
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, CV_BGRA2GRAY);
    else if (image.channels() == 1)
        gray = image;
}

double DistanceToCamera(double knownWidth, double focalLength, double perWidth) {
	return ((knownWidth * focalLength)/ perWidth);
}

int getPlayerOffsetX(int frame_count, float player_pos_x, float player_pos_y, int player_hgt)
{
	float numer, denom, res;
    if (player_hgt <= 50) {
		return (42);
    }
	else if (player_hgt > 50 && player_hgt <= 60) {
		numer = (float)(player_hgt - 50) * (75 - 42);
		denom = (float)(60 - 50);
		res = numer / denom;
		res += 42;
		if (res < 42) res = 42;
		return (res);	
	}
	else if (player_hgt > 60 && player_hgt <= 70) {
		numer = (float)(player_hgt - 60) * (125 - 75);
		denom = (float)(70 - 60);
		res = numer / denom;
		res += 75;
		return (res);	
	}	
	else if (player_hgt > 70 && player_hgt <= 80) {
		numer = (float)(player_hgt - 70) * (185 - 125);
		denom = (float)(80 - 70);
		res = numer / denom;
		res += 125;
		return (res);	
	}
	else if (player_hgt > 80 && player_hgt <= 90) {
		numer = (float) (player_hgt - 80) * (230 - 185);
		denom = (float) (90 - 80);
		res = numer / denom;
		res += 185;
		return (res);
	}
	else if (player_hgt > 90 && player_hgt <= 100) {
		numer = (float) (player_hgt - 90) * (266 - 230);
		denom = (float) (100 - 90);
		res = numer / denom;
		res += 230;
		return(res);
	}
	else {
		return (270.0);
	}
}

int getPlayerOffsetY(int frame_count, float player_pos_x, float player_pos_y, int player_hgt)
{
	float numer, denom, res;
    if (player_hgt <= 50) {
		//printf("%d:Height Range[... 50]  Expect Y Range[... 42] res=42 \n", frame_count);
		return (42);
    }
	else if (player_hgt > 50 && player_hgt <= 60) {
		numer = (float)(player_hgt - 50) * (75 - 42);
		denom = (float)(60 - 50);
		res = numer / denom;
		res += 42;
		if (res < 42) res = 42;
		//printf("%d:Height Range[50 ... 60]  Expect Y Range[42 ... 75] res=%f \n", frame_count, res);
		return (res);	
	}
	else if (player_hgt > 60 && player_hgt <= 70) {
		numer = (float)(player_hgt - 60) * (125 - 75);
		denom = (float)(70 - 60);
		res = numer / denom;
		res += 75;
		//printf("%d:Height Range[60 ... 70]  Expect Y Range[75 ... 125] res=%f \n", frame_count, res);
		return (res);	
	}	
	else if (player_hgt > 70 && player_hgt <= 80) {
		numer = (float)(player_hgt - 70) * (185 - 125);
		denom = (float)(80 - 70);
		res = numer / denom;
		res += 125;
		//printf("%d:Height Range[70 ... 80]  Expect Y Range[125 ... 185] res=%f \n", frame_count, res);
		return (res);	
	}
	else if (player_hgt > 80 && player_hgt <= 90) {
		numer = (float) (player_hgt - 80) * (230 - 185);
		denom = (float) (90 - 80);
		res = numer / denom;
		res += 185;
		//printf("%d:Height Range[80 ... 90]  Expect Y Range[185 ... 230] res=%f \n", frame_count, res);
		return (res);
	}
	else if (player_hgt > 90 && player_hgt <= 100) {
		numer = (float) (player_hgt - 90) * (266 - 230);
		denom = (float) (100 - 90);
		res = numer / denom;
		res += 230;
		//printf("%d:Height Range[90 ... 100]  Expect Y Range[230 ... 266] res=%f \n", frame_count, res);
		return(res);
	}
	else if (player_hgt > 100 && player_hgt <= 110) {
		numer = (float) (player_hgt - 90) * (266 - 230);
		denom = (float) (100 - 90);
		res = numer / denom;
		res += 230;
		//printf("%d:Height Range[90 ... 100]  Expect Y Range[230 ... 266] res=%f \n", frame_count, res);
		return(res);
	}
	else {
		res = 270.0;
		//printf("%d:Height Range[100 ...]  Expect Y Range[266 ..0] res=%f \n", frame_count, res);
		return (res);
	}
}

vector<Rect> get_sliding_windows(Mat& image,int winWidth,int winHeight)
{
  vector<Rect> rects;
  int step = 16;
  for(int i=0;i<image.rows;i+=step)
  {
      if((i+winHeight)>image.rows){break;}
      for(int j=0;j< image.cols;j+=step)    
      {
          if((j+winWidth)>image.cols){break;}
          Rect rect(j,i,winWidth,winHeight);
          rects.push_back(rect);
      }
  } 
  return rects;
}
Mat drawSemiCircle(Mat& image, int radius, Point center) {
	int temp1, temp2, temp3;
	int yval;
	
	for (int x=center.x-radius; x<=center.x+radius; x++) 
	{
		temp1 = radius * radius;
		temp2 = (x - center.x) * (x - center.x);
		temp3 = temp1 - temp2;
		yval = sqrt(temp3);
		yval += center.y;			
		circle(image, Point(x, yval), 1, Scalar(0,255,0), -1);			
	}
	return image;
}

/*
float euclideanDist(Point& p, Point& q) {
    Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}
*/

double euclideanDist(double x1, double y1, double x2, double y2)
{
	double x = x1 - x2; //calculating number to square in next step
	double y = y1 - y2;
	double dist;

	dist = pow(x, 2) + pow(y, 2);       //calculating Euclidean distance
	dist = sqrt(dist);                  

	return dist;
}

double oneDDist(double p1, double p2) {
	double dist;
	
	double p = p1 - p2;
	dist = pow(p, 2);
	dist = sqrt(dist);
	
	return dist;
}
int findIndex_BSearch(const vector< int> &numbersArray, int key) {

	int iteration = 0;
	int left = 0;
	int right = numbersArray.size()-1;
	int mid;

	while (left <= right) {
		iteration++;
		mid = (int) ((left + right) / 2);
		if (key <= numbersArray[mid]) 
		{
			right = mid - 1;
		}
		else if (key > numbersArray[mid])
		{
			left = mid + 1;
		}
	}
	return (mid);
}

