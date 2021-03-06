
#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <map>
#include <opencv2/core.hpp>

using namespace std;

struct LidarPoint
{ // single lidar point in space
    double x, y, z, r; // x,y,z in [m], r is point reflectivity
};


struct BoundingBox
{ // bounding box around a classified object (contains both 2D and 3D data)

    int boxID; // unique identifier for this bounding box
    int trackID; // unique identifier for the track to which this bounding box belongs

    cv::Rect roi; // 2D region-of-interest in image coordinates
    int classID; // ID based on class file provided to YOLO framework
    double confidence; // classification trust - confidence measure from YOLO step

    std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
};


struct DataFrame
{ // represents the available sensor information at the same time instance

    cv::Mat cameraImg; // camera image

    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
    std::vector<LidarPoint> lidarPoints;

    std::vector<BoundingBox> boundingBoxes; // ROI around detected objects in 2D image coordinates
    std::map<int, int> bbMatches; // bounding box matches between previous and current frame
};


struct Hyperparameters
{
    Hyperparameters()= default;

    string keypointDetector = "Shi_Tomasi"; // Shi_Tomasi, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    string descriptor = "BRIEF";                    // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
    string matcherType = "MAT_BF";                  // MAT_BF, MAT_FLANN
    string descriptorType = "DES_BINARY";           // DES_BINARY, DES_HOG
    string selectorType = "SEL_KNN";                // SEL_NN, SEL_KNN
};


struct KeypointMatchResult
{
    KeypointMatchResult()= default;
    std::pair<unsigned int, unsigned int> matchedImagePair = {0,0};
    unsigned int totalMatches = 0;
    unsigned int knnMatches = 0;
    unsigned int removed = 0;
    double percentageRemoved = 0.0;
};


struct KeypointCountResult
{
    KeypointCountResult()= default;
    string imageName = "No image name specified";
    unsigned int totalKeypoints = 0;
    double descriptorMatchingTime = 0.0;
    unsigned int precedingVehicleKeypoints = 0;
};


struct ResultLineItem
{
    ResultLineItem()= default;
    unsigned int frame = 0;
    double ttcLidar = 0.0;
    double ttcCamera = 0.0;
    unsigned int lidarPoints = 0;

    KeypointMatchResult keypointMatch;
    KeypointCountResult keypointCount;

    double descriptorExtractionTime = 0.0;
};


struct ResultSet
{
    ResultSet()= default;
    std::string detector = "";
    std::string descriptor = "";
    std::vector<ResultLineItem> data;
};


struct Experiment
{
    Experiment()= default;
    std::vector<ResultSet> resultSet;
    Hyperparameters hyperparameters;

    // Visualization and image saving options
    bool displayImageWindows = false;               // visualize matches between current and previous image?
    bool isFocusOnPrecedingVehicleOnly = true;      // only keep keypoints on the preceding vehicle?
    bool saveKeypointDetectionImagesToFile = false;  // save keypoint detection images to file
    bool saveKeypointMatchImagesToFile = true;      // save keypoint matching images to file
};

#endif /* dataStructures_h */
