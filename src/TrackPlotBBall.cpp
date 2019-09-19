#include "opencv2/dnn.hpp"
#include "opencv2/dnn/shape_utils.hpp"
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
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cvblob.h>


//#define IMG_DEBUG
#define SHOT_DEBUG

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace cvb;

class PlayerObs {
	public:
		PlayerObs();
	public:
		int		activeValue;
		int		radiusIdx;
		int 	placement;
		Point   position; 
		int 	frameCount;
		int 	shotsTaken;
		int		shotsMade;
};

PlayerObs::PlayerObs() : activeValue( 0 ), radiusIdx( 0 ), placement( 0 ), position(0, 0), frameCount( 0 ), shotsTaken(0), shotsMade(0)
{}


void getGray(const Mat& image, Mat& gray);
double DistanceToCamera(double knownWidth, double focalLength, double perWidth);
int getPlayerOffsetX(int frame_count, float player_pos_x, float player_pos_y, int player_hgt);
int getPlayerOffsetY(int frame_count, float player_pos_x, float player_pos_y, int player_hgt);
Mat drawSemiCircle(Mat& image, int radius, Point center);
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
	//int debugFlag = false;
	const string videoIdx 							= argc >= 2 ? argv[1] : "1";
	int fileNumber;
	string videofileName;

	bool sizeFlag = false;

	if ( argc > 1 ) {
		fileNumber = atoi( argv[1] );
	}
	else {
		fileNumber = 1;
	}
	stringstream vSS;
	vSS << fileNumber;
    string vIdx 									= vSS.str();
	if ( fileNumber <= 5 )
	{
		//videofileName 						= "/home/fred/Videos/testvideos/v" + vIdx + "-vga-fs.mp4";
		videofileName 						= "/home/fred/Videos/testvideos/v" + vIdx + ".mp4";
	}
	else if ( fileNumber > 5 && fileNumber <= 18 )
	{
		//videofileName 						= "/home/fred/Videos/testvideos/v" + vIdx + "-vga-fs.MOV";
		videofileName 						= "/home/fred/Videos/testvideos/v" + vIdx + ".MOV";
	}
	else if ( fileNumber > 18 && fileNumber <= 25)
	{
		//videofileName 						= "/home/fred/Videos/testvideos/v" + vIdx + "-vga-fs.mp4";
		videofileName 						= "/home/fred/Videos/testvideos/v" + vIdx + ".mp4";
	}
	else
	{
		videofileName 						= "/home/fred/Videos/testvideos/v" + vIdx + ".MOV";
	}

    help();
	int frameCount 									= 0;
	int frameWindow									= 0;
	const string bballPatternFile 					= "/home/fred/Pictures/OrgTrack_res/bball3_vga.jpg";
	Mat patternImage 								= imread(bballPatternFile);
	const string bballFileName 						= "/home/fred/Pictures/OrgTrack_res/bball-half-court-vga.jpeg";
    VideoCapture cap(videofileName);
    Mat bbsrc 										= imread(bballFileName);	
	int thresh 										= 85;
	RNG rng(12345);
	int newPlayerWindowSize 						= 50;
	PlayerObs newPlayerWindow;
	vector <int> radiusArray;
	Point courtArc[newPlayerWindowSize][1200];
	Mat threshold_output;
	vector<Vec4i> hierarchy;
	namedWindow("halfcourt", WINDOW_NORMAL);


	Ptr<BackgroundSubtractor> bg_model;
    bg_model 										= createBackgroundSubtractorMOG2(30, 16.0, false);
    Mat img;					//Source image from camera.  It may be scaled for efficiency reasons.
	Mat grayImage;				//Gray image of source image.
	Mat fgmask;					//Foreground mask image.
	Rect ballRect;				//Represents the box around the trackable basketball
	vector< vector<Point> > boardContours;	
	Scalar greenColor 								= Scalar (0, 215, 0);
	Scalar redColor 								= Scalar (0, 0, 215);
	Scalar blueColor								= Scalar (215, 0, 0);
	Scalar blackColor								= Scalar (40, 40, 40);
	Scalar orangeColor								= Scalar (0, 140, 255);
	Scalar aquaColor								= Scalar (255, 140, 0);
	Scalar rngColor = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
	bool haveBackboard 								= false;
    //vector<Rect> bodys;
	String body_cascade_name 						= "/home/fred/Pictures/OrgTrack_res/cascadeconfigs/haarcascade_fullbody.xml";
	String modelBinary								= "/home/fred/Dev/DNN_models/MadeShots/V1/made_8200.weights";
	String modelConfig								= "/home/fred/Dev/DNN_models/MadeShots/V1/made.cfg";
	CascadeClassifier body_cascade; 
	Mat firstFrame;
	bool semiCircleReady 							= false;
	Rect offsetBackboard;
	Rect Backboard;
	Point bodyPosit;
	Point bbCenterPosit;
	int BackboardCenterX;
	int BackboardCenterY;
	int leftActiveBoundary;
	int rightActiveBoundary;
	int topActiveBoundary;
	int bottomActiveBoundary;
	int leftBBRegionLimit;
	int rightBBRegionLimit;
	//int topBBRegionLimit;
	int bottomBBRegionLimit;
	const string OUTNAME = "v4_output_longversion.mp4";
	const string X_str = "x";
	const string O_str = "o";
	Point body_max_tl, body_max_br;
	Point body_tl, body_br;
	Point bodyCenter;


	CvTracks tracks;
	CvTracks body_tracks;
	Rect unionRect;

	if( !body_cascade.load( body_cascade_name ) )
	{
		printf("--(!)Error loading body_cascade_name\n"); return -1;
	}

	dnn::Net net = readNetFromDarknet(modelConfig, modelBinary);
	if (net.empty())
	{
		cout << "dnn model is empty." << endl;
		return -1;
	}

    vector<string> classNamesVec;
    ifstream classNamesFile("/home/fred/Dev/DNN_models/MadeShots/V1/made.names");
    if (classNamesFile.is_open())
    {
        string className = "";
        while (std::getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }

    if( !cap.isOpened() )
    {
		cout << "can not open video file " << videofileName << endl;
        return -1;
    }

	cap >> firstFrame;
	if (firstFrame.empty())
	{
        std::cout << "Cannot retrieve first video capture frame." << std::endl;
        return -1;
	}

    //int ex = static_cast<int>(cap.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
    Size S = Size((int) cap.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) cap.get(CAP_PROP_FRAME_HEIGHT));
    //cout << "S=" << S << endl;
	if (S.width > 640)
	{
		sizeFlag = true;
		S = Size(640,480);  //Size(320,240);  //Size(640, 480);
		resize(firstFrame, firstFrame, S);
		resize(bbsrc, bbsrc, S);
	}
	//Size outS = Size ((int) 2 * S.width, S.height);
	//VideoWriter outputVideo;
	//outputVideo.open(OUTNAME, ex, cap.get(CAP_PROP_FPS), outS, true);
	Mat finalImg(S.height, S.width+S.width, CV_8UC3);

	leftActiveBoundary 			= firstFrame.cols/4;  
	rightActiveBoundary			= firstFrame.cols*3/4;
	topActiveBoundary				= firstFrame.rows/4;
	bottomActiveBoundary			= firstFrame.rows*3/4;
	leftBBRegionLimit = (int) firstFrame.cols * 3 / 8;
	rightBBRegionLimit = (int) firstFrame.cols * 5 / 8;
	//topBBRegionLimit = (int) firstFrame.rows*2/8;
	bottomBBRegionLimit = (int) leftActiveBoundary;

	firstFrame.release();

	IplConvKernel* morphKernel = cvCreateStructuringElementEx(5, 5, 1, 1, CV_SHAPE_RECT, NULL);

	cv::Rect unionBodyRect;
	bool isFirstPass = true;

    for(;;)
    {
        cap >> img;

        if (sizeFlag)
        	resize(img, img, S);

        if (fileNumber > 10 && fileNumber <= 21)
        {
        	flip(img, img, 0);
        }

		frameCount++;
		
        if( img.empty() )
            break;

		//if( !body_cascade.load( body_cascade_name ) )
		//{
		//	printf("--(!)Error loading body_cascade_name\n"); return -1;
		//}

		stringstream frame_ss;
		if (!haveBackboard)
		{
			getGray(img,grayImage);											//Converts to a gray image.  All we need is a gray image for cv computing.
			Canny(grayImage, grayImage, thresh, thresh*2, 3);
			findContours( grayImage, boardContours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );
			vector< vector<Point> > contours_poly( boardContours.size() );
			vector<Rect> boundRect( boardContours.size() );

			for ( size_t i = 0; i < boardContours.size(); i++ )
			{
				approxPolyDP(Mat(boardContours[i]),contours_poly[i],3,true);
				boundRect[i] = boundingRect(Mat(contours_poly[i]));

				double bb_w = (double) boundRect[i].size().width;
				double bb_h = (double) boundRect[i].size().height;
        		double bb_ratio = (double) bb_w / bb_h;
        		if ( (boundRect[i].x > leftBBRegionLimit)
					  && (boundRect[i].x < rightBBRegionLimit)
					  && (boundRect[i].x + boundRect[i].width < rightBBRegionLimit)
					  && (boundRect[i].y < bottomBBRegionLimit)
					  && (boundRect[i].area() > 50)
					  && (bb_ratio < 1.3)
					  && (bb_w > (bb_h * 0.74) ) )
				{
        			if (isFirstPass)
        			{
        				unionRect = boundRect[i];
        				isFirstPass = false;
        			}
        			else if (frameCount < 100)
        			{
        				unionRect = unionRect | boundRect[i];
        			}

        			if (frameCount > 99) {
						offsetBackboard = Rect(unionRect.tl().x,
												unionRect.tl().y - 40,
												unionRect.size().width,
                                                unionRect.size().height);

						//Rect offsetBackboard2 = Rect(unionRect.tl().x + 33,
						//							unionRect.tl().y - 35,
						//							unionRect.size().width,
						//							unionRect.size().height);



						Point semiCircleCenterPt( (offsetBackboard.tl().x+offsetBackboard.width/2) , (offsetBackboard.tl().y + offsetBackboard.height/2) );

		                bbCenterPosit = semiCircleCenterPt;  // Coords for halfcourt shot chart image
        				Backboard = unionRect;   //Coords for true video content

        				BackboardCenterX = (Backboard.tl().x+(Backboard.width*3/4));   // (Backboard.tl().x+(Backboard.width/2));
        				BackboardCenterY = (Backboard.tl().y+(Backboard.height*3/4));    // (Backboard.tl().y+(Backboard.height/2));
        				//BackboardCenterX = (offsetBackboard2.tl().x + (offsetBackboard2.width/2));
        				//BackboardCenterY = (offsetBackboard2.tl().y + (offsetBackboard2.height/2));
        				haveBackboard = true;
        			}

				}  //(boundRect[i].x > leftBBRegionLimit)

			}  //( size_t i = 0; i < boardContours.size(); i++ )

		}   //if (!haveBackboard)

		///*************************End of main code to detect BackBoard*************************

		///*******Start of main code to detect Basketball*************************
		frame_ss << frameCount;
		string frame_str = frame_ss.str();
		if (haveBackboard)
		{
			putText(bbsrc, "C",	bbCenterPosit,	FONT_HERSHEY_PLAIN, 1,	greenColor, 1, 0.5);

	    	if (!semiCircleReady)
	    	{
	    		int radiusIdx = 0;
	    		for (int radius = 40; radius < 280; radius += 20)   //Radius for euclidDistFromBB
	    		{
	    			radiusArray.push_back(radius);

	    			int temp1, temp2, temp3;
	    			int yval;
	    			for (int j = (bbCenterPosit.x - radius); j <= (bbCenterPosit.x + radius); j++)   //Using Pythagorean's theorem to find positions on each court arc.
	    			{
	    				temp1 = radius * radius;
	    				temp2 = (j - bbCenterPosit.x) * (j - bbCenterPosit.x);
	    				temp3 = temp1 - temp2;
	    				yval = sqrt(temp3);
	    				yval += bbCenterPosit.y;
	    				Point ptTemp = Point(j, yval);
	    				courtArc[radiusIdx][j] = ptTemp;
	    				circle(bbsrc, ptTemp, 1, blueColor, -1);
	    			}

	    			radiusIdx++;
	    		}
	    		semiCircleReady = true;
	    	}

			IplImage iImg = img;
			IplImage *segmentated = cvCreateImage(S, 8, 1);

			unsigned int S_height = (unsigned int) S.height;
		    unsigned int S_width = (unsigned int) S.width;
			for (unsigned int j=0; j< S_height; j++)
			{
				for (unsigned int i=0; i<S_width; i++)
				{
					CvScalar c = cvGet2D(&iImg, j, i);

					double b = ((double)c.val[0])/255.;
					double g = ((double)c.val[1])/255.;
					double r = ((double)c.val[2])/255.;

					unsigned char f = 255 * ( ( r > 0.12 + g ) && ( r > 0.16 + b ) && (g > 0.013 + b));   //Yes good for now!

					cvSet2D(segmentated, j, i, CV_RGB(f, f, f));
				}
			}

			cvMorphologyEx(segmentated, segmentated, NULL, morphKernel, CV_MOP_OPEN, 1);

			IplImage *labelImg = cvCreateImage(cvGetSize(&iImg), IPL_DEPTH_LABEL, 1);

			CvBlobs blobs;
			unsigned int totalLabeledPixels = cvLabel(segmentated, labelImg, blobs);

			cvFilterByArea(blobs, 75/*25*/, 1000000); //25, 1000);
			//cvRenderBlobs(labelImg, blobs, &iImg, &iImg, CV_BLOB_RENDER_BOUNDING_BOX);
			cvUpdateTracks(blobs, tracks, 2., 10, 2);

			unsigned left = 0, top = 0;
			unsigned t_width = 0, t_height = 0;

			//vector<cv::Rect> trRects;
			//for (CvTracks::const_iterator jt = tracks.begin(); jt!=tracks.end(); ++jt)
			for (CvBlobs::const_iterator jt = blobs.begin(); jt!=blobs.end(); ++jt)
			{
			  left = jt->second->minx;
			  top = jt->second->miny;
			  t_width = jt->second->maxx - jt->second->minx;
			  t_height = jt->second->maxy - jt->second->miny;
			  ballRect = cv::Rect(left, top, t_width, t_height);
			  rectangle(img, ballRect.tl(), ballRect.br(), orangeColor, 1, 8, 0);
			  //trRects.push_back(localRect);
			  //rectangle(img, localRect.tl(), localRect.br(), redColor, 1, 8, 0);
///			}  // for (CvTracks::const_iterator jt)

			//cvRenderTracks(tracks, &iImg, &iImg, CV_TRACK_RENDER_ID|CV_TRACK_RENDER_BOUNDING_BOX);

			//------------Track the basketball!!!!---------------

			//Writes output to output array (basketballTracker)

			///unsigned int nTracks =  trRects.size();  //tracks.size();
			///if (nTracks > 0)
			///{
///				for (vector<cv::Rect>::iterator jt = trRects.begin(); jt!=trRects.end(); ++jt)
///				{
					//ballRect = *jt;


					if ( (ballRect.x > leftActiveBoundary)
									&& (ballRect.x < rightActiveBoundary)
									&& (ballRect.y > topActiveBoundary)
									&& (ballRect.y < bottomActiveBoundary) )
					{
						//The basketball on video frames.
						Rect objIntersect = Backboard & ballRect;

						//---Start of the process of identifying a shot at the basket!!!------------
						if (objIntersect.area() > 0)
						{
							//putText(bbsrc, X_str, courtArc[newPlayerWindow.radiusIdx][newPlayerWindow.placement], FONT_HERSHEY_PLAIN, 1 , redColor, 1, LINE_4);   //, 2);

							//Predict a made shot
							Mat basketRoI = img(Backboard).clone();
				            //resize(basketRoI, basketRoI, Size(416,416));

				            //! [Prepare blob]
				            Mat inputBlob = blobFromImage(basketRoI, 1 / 255.F, Size(416, 416), Scalar(), true, false); //Convert Mat to batch of images
				            //! [Prepare blob]

				            //! [Set input blob]
				            net.setInput(inputBlob, "data");                   //set the network input
				            //! [Set input blob]

				            //! [Make forward pass]
				            Mat detectionMat = net.forward("detection_out");   //compute output

				            for (int i = 0; i < detectionMat.rows; i++)
				            {
				                const int probability_index = 5;
				                const int probability_size = detectionMat.cols - probability_index;
				                float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);

				                size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
				                float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

				                if (confidence > 0.24)
				                {
				                    float x = detectionMat.at<float>(i, 0);
				                    float y = detectionMat.at<float>(i, 1);
				                    float width = detectionMat.at<float>(i, 2);
				                    float height = detectionMat.at<float>(i, 3);
				                    int xLeftBottom = static_cast<int>((x - width / 2) * basketRoI.cols);
				                    int yLeftBottom = static_cast<int>((y - height / 2) * basketRoI.rows);
				                    int xRightTop = static_cast<int>((x + width / 2) * basketRoI.cols);
				                    int yRightTop = static_cast<int>((y + height / 2) * basketRoI.rows);

				                    Rect object(xLeftBottom, yLeftBottom,
				                                xRightTop - xLeftBottom,
				                                yRightTop - yLeftBottom);

				                    if (objectClass < classNamesVec.size())
				                    {
				                        frame_ss.str("");
				                        frame_ss << confidence;
				                        String conf(frame_ss.str());
				                        String label = String(classNamesVec[objectClass]) + ": " + conf;
				                        int baseLine = 0;
				                        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				                        //rectangle(basketRoI, Rect(Point(xLeftBottom, yLeftBottom ),
				                        //                      Size(labelSize.width, labelSize.height + baseLine)),
				                        //          Scalar(255, 255, 255), CV_FILLED);
				                        //putText(basketRoI, label, Point(xLeftBottom, yLeftBottom+labelSize.height),
				                        //        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
				                    }
				                    else
				                    {
				                        cout << "Class: " << objectClass << endl;
				                        cout << "Confidence: " << confidence << endl;
				                        cout << " " << xLeftBottom
				                             << " " << yLeftBottom
				                             << " " << xRightTop
				                             << " " << yRightTop << endl;
				                    }
				                }
				            }
				            //imshow("YOLO: Detections", basketRoI);
				            //********End of Shot Prediction **********

							//---Start of using player position on halfcourt image to draw shot location-----

							if (haveBackboard && frameCount > frameWindow)
							{
								frameWindow = frameCount + 10;
								//circle(bbsrc, courtArc[newPlayerWindow.radiusIdx][newPlayerWindow.placement], 1, Scalar(0, 165, 255), 3);
								//putText(bbsrc, O_str, courtArc[newPlayerWindow.radiusIdx][newPlayerWindow.placement], FONT_HERSHEY_PLAIN, 1 , greenColor, 1, LINE_4);   //, 2);
								Point offPt = Point(courtArc[newPlayerWindow.radiusIdx][newPlayerWindow.placement].x-5, courtArc[newPlayerWindow.radiusIdx][newPlayerWindow.placement].y-5);
								putText(bbsrc, frame_str, offPt, FONT_HERSHEY_PLAIN, 1 , rngColor, 1, 0.5);   //, 2);
								//putText(bbsrc, X_str, courtArc[newPlayerWindow.radiusIdx][newPlayerWindow.placement], FONT_HERSHEY_PLAIN, 1 , redColor, 1, LINE_4);   //, 2);
							}
						}
						//---Start of using player position on halfcourt image to draw shot location-----
						//---End of the process of identifying a shot at the basket!!!------------
					}
///				}  //for (vector<cv::Rect>::iterator jt)
			}  // for (CvTracks::const_iterator jt)
				///*******End of code to detect & select Basketball*************************
			///}  //if (nTracks > 0)

			//-- detect body
			getGray(img,grayImage);			//Converts to a gray and blur image.  All we need is a gray image for cv computing.
			bg_model->apply(grayImage, fgmask);
			//imshow("fgmask", fgmask);

			IplImage *body_segmentated = new IplImage(fgmask);
			//cvMorphologyEx(body_segmentated, body_segmentated, NULL, morphKernel, CV_MOP_OPEN, 1);
			IplImage* body_labelImg = cvCreateImage(S, IPL_DEPTH_LABEL, 1);
			CvBlobs body_blobs;
			unsigned int body_result = cvLabel(body_segmentated, body_labelImg, body_blobs);
			//cvShowImage("body_segmentated", body_segmentated);
			//cvFilterByArea(body_blobs, 250, 20000);
			//cvRenderBlobs(body_labelImg, body_blobs, &iImg, &iImg, CV_BLOB_RENDER_BOUNDING_BOX);
			cvUpdateTracks(body_blobs, body_tracks, 200., 1, 1);
			//cvRenderTracks(body_tracks, &iImg, &iImg, CV_TRACK_RENDER_ID|CV_TRACK_RENDER_BOUNDING_BOX);

			bool firstRect = true;
			vector<cv::Rect> body_trRects;
			//for (CvTracks::const_iterator jt = body_tracks.begin(); jt!=body_tracks.end(); ++jt)
			for (CvBlobs::const_iterator jt = body_blobs.begin(); jt!=body_blobs.end(); ++jt)
			{
			  left = jt->second->minx;
			  top = jt->second->miny;
			  t_width = jt->second->maxx - jt->second->minx;
			  t_height = jt->second->maxy - jt->second->miny;
			  float f_width = (float) t_width;
			  float f_height = (float) t_height;
			  cv::Rect localRect = cv::Rect(left, top, t_width, t_height);
			  if ( f_height > (f_width*1.5) )
			  {
				  if (firstRect)
				  {
					  unionBodyRect = localRect;
					  firstRect = false;
				  }
				  else
				  {
					  unionBodyRect = unionBodyRect | localRect;
				  }
				  body_trRects.push_back(localRect);
			  }
			}


//			if ((float) unionBodyRect.height < (float) (unionBodyRect.width*1.5))
//				continue;

//			if ( (unionBodyRect.height < 50) || (unionBodyRect.height > 275) )
//				continue;

			cout << frameCount << " : unionBodyRect=" << unionBodyRect << endl;
			rectangle(img, unionBodyRect.tl(), unionBodyRect.br(), greenColor, 2, 8, 0);

///			int x_scale = 15;  int y_scale = 75;
///			body_cascade.detectMultiScale(grayImage, bodys, 1.02, 1, CV_HAAR_DO_CANNY_PRUNING, Size(x_scale, y_scale));



			////for( int j = 0; j < (int) body_trRects.size(); j++ )
			////{
				Point bodyCenter( unionBodyRect.x + unionBodyRect.width*0.5, unionBodyRect.y + unionBodyRect.height*0.5 );
				//-----------Identifying player height and position!!--------------
				Point ballCenter( (ballRect.x + ballRect.width* 0.5), (ballRect.y + ballRect.height * 0.5));

				double ballBodyDistance = euclideanDist(bodyCenter.x, bodyCenter.y, ballCenter.x, ballCenter.y);

				if (unionBodyRect.width > 100 || unionBodyRect.height > 350)
					continue;

				if (unionBodyRect.br().y < Backboard.br().y)
				{
//					cout << " false body by Backboard" << endl;
					continue;
				}

				Rect bodyRect = Rect(unionBodyRect.tl(), unionBodyRect.br());
				Rect badBodyIntersect = Backboard & bodyRect;
				if (badBodyIntersect.area() > (0.6 * bodyRect.area()))
				{
//					cout << " false body2 by Backboard" << endl;
					continue;
				}

				//--- Start of adjusting player position on image of half court!!!-----
				newPlayerWindow.frameCount = frameCount;
				newPlayerWindow.activeValue = 1;
				newPlayerWindow.position = bodyCenter;

				double euclidDistFromBB = euclideanDist((double) BackboardCenterX,
												  (double) BackboardCenterY,
												  (double) bodyCenter.x,
												  (double) bodyCenter.y);

				double xDistFromBB = oneDDist(BackboardCenterX, bodyCenter.x);
				double yDistFromBB = oneDDist(BackboardCenterY, bodyCenter.y);

				float heightNorm = (float) unionBodyRect.height / 350.0f;

				////ellipse( img, bodyCenter, Size( unionBodyRect.width*0.5, unionBodyRect.height*0.5), 0, 0, 360, aquaColor, 4, 8, 0 );
				line(img, Point(BackboardCenterX, BackboardCenterY), bodyCenter, blueColor, 1, 8);

				int eu_mid_x = (int) (BackboardCenterX + bodyCenter.x) * 0.5;
				int eu_mid_y = (int) (BackboardCenterY + bodyCenter.y) * 0.5;
				stringstream eu_ss;
				eu_ss << euclidDistFromBB;
				string eu_str = eu_ss.str();
				putText(img, eu_str,Point(eu_mid_x - 5, eu_mid_y - 5), FONT_HERSHEY_PLAIN, 1, blueColor, 1, 0.5);


				//stringstream idx_ss;
				//idx_ss << j;
				//string idx_str = idx_ss.str();
				//putText(img, idx_str,Point((int)unionBodyRect.x + unionBodyRect.width*0.5-5, (int)unionBodyRect.y-5), FONT_HERSHEY_PLAIN, 1, aquaColor, 1, 0.5);

				stringstream hss;
				hss << unionBodyRect.height;
				string bodyHgt_str = hss.str();
				putText(img, bodyHgt_str,Point((int)unionBodyRect.x + unionBodyRect.width*0.5-5, (int)unionBodyRect.y + unionBodyRect.height*0.5-5), FONT_HERSHEY_PLAIN, 1, greenColor, 1, 0.5);


				if (euclidDistFromBB > 250)
				{
					//cout << frameCount <<": distBB 135+    heightNorm=" << heightNorm << endl;

					newPlayerWindow.radiusIdx = radiusArray.size() * 0.99;
					//euclidDistFromBB += 120;

					int tempPlacement = (bbCenterPosit.x + radiusArray[newPlayerWindow.radiusIdx])
									    - (bbCenterPosit.x - radiusArray[newPlayerWindow.radiusIdx]);

					if (bodyCenter.x > BackboardCenterX)
						tempPlacement -= 1;
					else
						tempPlacement = 0;

					tempPlacement += (bbCenterPosit.x - radiusArray[newPlayerWindow.radiusIdx]);

					newPlayerWindow.placement = tempPlacement;
				}
				else if (euclidDistFromBB < 100)
				{
					//cout << frameCount <<": distBB -30 heightNorm=" << heightNorm << endl;

					int tempPlacement;
					if (bodyCenter.x < BackboardCenterX)
						tempPlacement = 0;
					else
						tempPlacement = (bbCenterPosit.x + radiusArray[newPlayerWindow.radiusIdx])
									- (bbCenterPosit.x - radiusArray[newPlayerWindow.radiusIdx]) - 1;

					newPlayerWindow.placement = tempPlacement;
					newPlayerWindow.radiusIdx = radiusArray.size() * 0.01;
				}
				else
				{
					//cout << frameCount <<": ELSE  heightNorm=" << heightNorm << endl;

					if (heightNorm < 0.5)    //NOTE:  If not true, then we have inaccurate calculation of body height from detectMultiscale method.  Do not estimate a player position for it.
					{
						newPlayerWindow.radiusIdx = findIndex_BSearch(radiusArray, euclidDistFromBB);
						newPlayerWindow.radiusIdx += 5;

						if ((xDistFromBB < 51) && (yDistFromBB < 70))
							newPlayerWindow.radiusIdx = 0;

						double percentPlacement = (double) (bodyCenter.x - leftActiveBoundary) / (rightActiveBoundary - leftActiveBoundary);
						int leftRingBound		= bbCenterPosit.x - radiusArray[newPlayerWindow.radiusIdx];
						int rightRingBound		= bbCenterPosit.x + radiusArray[newPlayerWindow.radiusIdx];
						int chartPlacementTemp	= (rightRingBound - leftRingBound) * percentPlacement;
						int chartPlacement		= leftRingBound + chartPlacementTemp;

						newPlayerWindow.placement = chartPlacement;
					}
				}
				//--- End of adjusting player position on image of half court!!!-----
			////}

			rectangle(img, Backboard.tl(), Backboard.br(), redColor, 2, 8, 0);
			circle(img, Point(BackboardCenterX, BackboardCenterY), 1, greenColor, 3);

			stringstream iss;
			iss << frameCount;
			string i_str = iss.str();
			putText(img, frame_str, Point(5, 20), FONT_HERSHEY_PLAIN, 2 , greenColor, 0.5);
			imwrite("/home/fred/Temp/Temp/v23_stills/image" + i_str + ".jpg", img);

			cvReleaseImage(&labelImg);
			cvReleaseImage(&segmentated);
			//trRects.clear();
		}  //if (haveBackboard)

		//Create string of frame counter to display on video window.
		Mat left(finalImg, Rect(0, 0, img.cols, img.rows));
		img.copyTo(left);
		Mat right(finalImg, Rect(bbsrc.cols, 0, bbsrc.cols, bbsrc.rows));
		bbsrc.copyTo(right);		

		imshow("halfcourt", finalImg); //bbsrc);
        //imshow("image", img);

        char k = (char)waitKey(30);
        if( k == 27 ) break;

		//outputVideo << finalImg;
    }

//endloop:
//	cout << "Debug flag ends processing" << endl;
    cvReleaseStructuringElement(&morphKernel);

    return 0;
}


//Blurs, i.e. smooths, an image using the normalized box filter.  Used to reduce noise.
void getGray(const Mat& image, Mat& gray)
{
    if (image.channels()  == 3)
        cv::cvtColor(image, gray, COLOR_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, COLOR_BGRA2GRAY);
    else if (image.channels() == 1)
        gray = image;

    blur(gray, gray, Size(3,3));
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

