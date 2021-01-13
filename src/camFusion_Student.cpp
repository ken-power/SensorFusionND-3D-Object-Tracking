
#include <iostream>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> & boundingBoxes,
                         std::vector<LidarPoint> & lidarPoints,
                         float shrinkFactor,
                         cv::Mat & P_rect_xx,
                         cv::Mat & R_rect_xx,
                         cv::Mat & RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for(auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for(vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if(smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if(enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> & boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for(auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int) it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for(size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    if(false)
    {
        cv::namedWindow(windowName, 1);
        cv::imshow(windowName, topviewImg);
    }

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }

    static int imageCount = 0;
    string outputFilename =
            string("../results/images/lidar_top_view/") + string("image_") + to_string(imageCount++) + string(".png");
    cv::imwrite(outputFilename, topviewImg);
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox & boundingBox,
                              std::vector<cv::KeyPoint> & kptsPrev,
                              std::vector<cv::KeyPoint> & kptsCurr,
                              std::vector<cv::DMatch> & kptMatches)
{
    std::vector<double> euclideanDistance;

    for(auto it = kptMatches.begin(); it != kptMatches.end(); it++)
    {
        int currKptIndex = (*it).trainIdx;
        const auto & currKeyPoint = kptsCurr[currKptIndex];

        if(boundingBox.roi.contains(currKeyPoint.pt))
        {
            int prevKptIndex = (*it).queryIdx;
            const auto & prevKeyPoint = kptsPrev[prevKptIndex];

            euclideanDistance.push_back(cv::norm(currKeyPoint.pt - prevKeyPoint.pt));
        }
    }

    int pair_num = euclideanDistance.size();
    double euclideanDistanceMean = std::accumulate(euclideanDistance.begin(), euclideanDistance.end(), 0.0) / pair_num;

    for(auto it = kptMatches.begin(); it != kptMatches.end(); it++)
    {
        int currKptIndex = (*it).trainIdx;
        const auto & currKeyPoint = kptsCurr[currKptIndex];

        if(boundingBox.roi.contains(currKeyPoint.pt))
        {
            int prevKptIndex = (*it).queryIdx;
            const auto & prevKeyPoint = kptsPrev[prevKptIndex];

            double temp = cv::norm(currKeyPoint.pt - prevKeyPoint.pt);

            double euclideanDistanceMean_Augment = euclideanDistanceMean * 1.3;
            if(temp < euclideanDistanceMean_Augment)
            {
                boundingBox.keypoints.push_back(currKeyPoint);
                boundingBox.kptMatches.push_back(*it);
            }
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> & kptsPrev,
                      std::vector<cv::KeyPoint> & kptsCurr,
                      std::vector<cv::DMatch> kptMatches,
                      double frameRate,
                      double & TTC,
                      cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame

    for(auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for(auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if(distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if(distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    //double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0
                                                     : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}


void computeTTCLidar(std::vector<LidarPoint> & lidarPointsPreviousFrame,
                     std::vector<LidarPoint> & lidarPointsCurrentFrame,
                     double frameRate,
                     double & TTC)
{
    // Sort ascending on the x coordinate only
    std::sort(lidarPointsPreviousFrame.begin(), lidarPointsPreviousFrame.end(), [](LidarPoint a, LidarPoint b) {
                  return a.x < b.x;
              }
    );

    std::sort(lidarPointsCurrentFrame.begin(), lidarPointsCurrentFrame.end(), [](LidarPoint a, LidarPoint b) {
                  return a.x < b.x;
              }
    );

    /**
     Using the constant-velocity model (as opposed to a constant-acceleration model)
     TTC = d1 * delta_t / (d0 - d1)
     where: d0 is the previous frame's closing distance (front-to-rear bumper)
            d1 is the current frame's closing distance (front-to-rear bumper)
            delta_t is the time elapsed between images (1 / frameRate)
     Note: this function does not take into account the distance from the lidar origin to the front bumper of our vehicle.
     It also does not account for the curvature or protrusions from the rear bumper of the preceding vehicle.
    */
    double d0 = lidarPointsPreviousFrame[lidarPointsPreviousFrame.size() / 2].x;
    double d1 = lidarPointsCurrentFrame[lidarPointsCurrentFrame.size() / 2].x;
    double delta_t = 1.0 / frameRate;

    TTC = d1 * delta_t / (d0 - d1);
}


void matchBoundingBoxes(std::vector<cv::DMatch> & matches,
                        std::map<int, int> & boundingBoxBestMatches,
                        DataFrame & previousFrame,
                        DataFrame & currentFrame)
{
    std::multimap<int, int> boundingBoxMatches{};  // A pair of IDs to track bounding boxes

    for(const auto & match : matches)
    {
        cv::KeyPoint keypointsPreviousFrame = previousFrame.keypoints[match.queryIdx];
        cv::KeyPoint keypointsCurrentFrame = currentFrame.keypoints[match.trainIdx];

        unsigned int boxIdPreviousFrame;
        unsigned int boxIdCurrentFrame;

        for(const auto & boundingBox : previousFrame.boundingBoxes)
        {
            if(boundingBox.roi.contains(keypointsPreviousFrame.pt))
            {
                boxIdPreviousFrame = boundingBox.boxID;
            }
        }

        for(const auto & boundingBox : currentFrame.boundingBoxes)
        {
            if(boundingBox.roi.contains(keypointsCurrentFrame.pt))
            {
                boxIdCurrentFrame = boundingBox.boxID;
            }
        }

        boundingBoxMatches.insert({boxIdCurrentFrame, boxIdPreviousFrame});
    }

    vector<int> boundingBoxIdsCurrentFrame{};

    for(const auto & boundingBox : currentFrame.boundingBoxes)
    {
        boundingBoxIdsCurrentFrame.push_back(boundingBox.boxID);
    }

    for(const int boxIdCurrentFrame : boundingBoxIdsCurrentFrame)
    {
        auto it = boundingBoxMatches.equal_range(boxIdCurrentFrame);
        unordered_map<int, int> boundingBoxIdMatches;
        for(auto itr = it.first; itr != it.second; ++itr)
        {
            boundingBoxIdMatches[itr->second]++;
        }

        // find the max frequency
        unsigned int maxBoxID = 0;
        int matchingBoxID = -1;

        for(const auto & boxIdMatch : boundingBoxIdMatches)
        {
            if(maxBoxID < boxIdMatch.second)
            {
                matchingBoxID = boxIdMatch.first;
                maxBoxID = boxIdMatch.second;
            }
        }

        boundingBoxBestMatches.insert({matchingBoxID, boxIdCurrentFrame});
    }
}
