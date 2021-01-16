
#ifndef camFusion_hpp
#define camFusion_hpp

#include <stdio.h>
#include <vector>
#include <opencv2/core.hpp>
#include "dataStructures.h"


void clusterLidarWithROI(std::vector<BoundingBox> & boundingBoxes,
                         std::vector<LidarPoint> & lidarPoints,
                         float shrinkFactor,
                         cv::Mat & P_rect_xx,
                         cv::Mat & R_rect_xx,
                         cv::Mat & RT);

void clusterKptMatchesWithROI(BoundingBox & boundingBox,
                              std::vector<cv::KeyPoint> & keypointsPreviousFrame,
                              std::vector<cv::KeyPoint> & keypointsCurrentFrame,
                              std::vector<cv::DMatch> & keypointMatches);

void matchBoundingBoxes(std::vector<cv::DMatch> & matches,
                        std::map<int, int> & boundingBoxBestMatches,
                        DataFrame & previousFrame,
                        DataFrame & currentFrame);

void show3DObjects(vector<BoundingBox> & boundingBoxes,
                   cv::Size worldSize,
                   cv::Size imageSize,
                   bool bWait,
                   ResultLineItem & result,
                   const string & detector,
                   const string & descriptor);

void computeTTCCamera(std::vector<cv::KeyPoint> & kptsPrev, std::vector<cv::KeyPoint> & kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double & TTC, cv::Mat *visImg = nullptr);

void computeTTCLidar(std::vector<LidarPoint> & lidarPointsPreviousFrame,
                     std::vector<LidarPoint> & lidarPointsCurrentFrame, double frameRate, double & TTC);

const string GetTtcFilename(const string detector, const string descriptor, const int frame);
const string GetLidarFilename(const string detector, const string descriptor, const int frame);

#endif /* camFusion_hpp */
