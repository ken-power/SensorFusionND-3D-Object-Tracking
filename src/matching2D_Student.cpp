#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(vector<cv::KeyPoint> &kPtsSource,
                      vector<cv::KeyPoint> &kPtsRef,
                      cv::Mat &descSource,
                      cv::Mat &descRef,
                      vector<cv::DMatch> &matches,
                      const string &descriptorType,
                      const string& matcherType,
                      const string& selectorType,
                      ResultLineItem &result)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType == "MAT_BF")  // Brute Force matching
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType == "MAT_FLANN")
    {
        if (descSource.type() != CV_32F)
        {
            // OpenCV bug workaround : convert binary descriptors to floating point due
            // to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        // implement FLANN matching
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType == "SEL_NN")
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType == "SEL_KNN")
    {
        // implement k-nearest-neighbor matching
        vector<vector<cv::DMatch>> knn_matches;
        auto t = (double) cv::getTickCount();

        int k = 2; // finds the 2 best matches
        matcher->knnMatch(descSource, descRef, knn_matches, k);

        t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms";

        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;

        for (auto &knn_match : knn_matches)
        {
            if (knn_match[0].distance < minDescDistRatio * knn_match[1].distance)
            {
                matches.push_back(knn_match[0]);
            }
        }

        cout << "; matches = " << matches.size() << ", KNN matches = " << knn_matches.size();

        long keypointsRemoved = static_cast<long>(knn_matches.size() - matches.size());
        float percentageKeypointsRemoved =
                (static_cast<float>(keypointsRemoved) / static_cast<float>(knn_matches.size())) * 100;

        cout << " => keypoints removed = " << keypointsRemoved << " (" << percentageKeypointsRemoved << "%)" << endl;

        result.keypointMatch.totalMatches = matches.size();
        result.keypointMatch.knnMatches = knn_matches.size();
        result.keypointMatch.removed = keypointsRemoved;
        result.keypointMatch.percentageRemoved = percentageKeypointsRemoved;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, const string& descriptorType, ResultLineItem &result)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;

    if (descriptorType == "BRISK")
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType == "BRIEF")
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType == "ORB")
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType == "FREAK")
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType == "AKAZE")
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType == "SIFT")
    {
        extractor = cv::SIFT::create();
    }
    else
    {
        std::cerr << descriptorType << " is not a supported descriptor type." << endl;
    }

    // perform feature description
    auto t = (double)cv::getTickCount();

    try
    {
        extractor->compute(img, keypoints, descriptors);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Exception with Detector/Descriptor combination: " << ex.what() << std::endl;
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    double extractionTime = 1000 * t / 1.0;
    cout << descriptorType << " descriptor extraction in " << extractionTime << " ms" << endl;
    result.descriptorExtractionTime = extractionTime;
}

// Detect keypoints in image using the traditional Shi-Tomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool displayImageWindows, bool saveImageFiles, ResultLineItem &result)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative co-variation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    auto t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto & corner : corners)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f(corner.x, corner.y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    double duration = 1000 * t / 1.0;
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << duration << " ms" << endl;

    result.keypointCount.totalKeypoints = keypoints.size();
    result.keypointCount.descriptorMatchingTime = duration;

    visualizeKeypoints(keypoints, img, "Shi_Tomasi_Corner_Detection_Results", displayImageWindows, saveImageFiles, result);
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool displayImageWindows, bool saveImageFiles, ResultLineItem &result)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    auto t = (double)cv::getTickCount();
    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);


    // Locate local maxima in the Harris response matrix
    // and perform a non-maximum suppression (NMS) in a local neighborhood around
    // each maximum. The resulting coordinates shall be stored in a list of keypoints
    // of the type `vector<cv::KeyPoint>`.
    // Look for prominent corners and instantiate keypoints
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression

    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int) dst_norm.at<float>(j, i);
            if (response > minResponse) // only store points above a threshold
            {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto & keypoint : keypoints)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, keypoint);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > keypoint.response)
                        {  // if overlap is >t AND response is higher for new kpt
                            keypoint = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap) // only add new key point if no overlap has been found in previous NMS
                {
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    double duration = 1000 * t / 1.0;
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << duration << " ms" << endl;

    result.keypointCount.totalKeypoints = keypoints.size();
    result.keypointCount.descriptorMatchingTime = duration;

    visualizeKeypoints(keypoints, img, "Harris_Corner_Detection_Results", displayImageWindows, saveImageFiles, result);
}


void detKeypointsFAST(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool displayImageWindows, bool saveImageFiles, ResultLineItem &result)
{
    // difference between intensity of the central pixel and pixels of a circle around this pixel
    int threshold = 30;

    // perform non-maxima suppression on keypoints
    bool bNMS = true;

    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
    cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(threshold, bNMS, type);

    detectKeypoints(detector, "FAST", keypoints, img, displayImageWindows, saveImageFiles, result);
}

void detKeypointsBRISK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool displayImageWindows, bool saveImageFiles, ResultLineItem &result)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();

    detectKeypoints(detector, "BRISK", keypoints, img, displayImageWindows, saveImageFiles, result);
}

void detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool displayImageWindows, bool saveImageFiles, ResultLineItem &result)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();

    detectKeypoints(detector, "ORB", keypoints, img, displayImageWindows, saveImageFiles, result);
}

void detKeypointsAKAZE(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool displayImageWindows, bool saveImageFiles, ResultLineItem &result)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();

    detectKeypoints(detector, "AKAZE", keypoints, img, displayImageWindows, saveImageFiles, result);
}

void detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool displayImageWindows, bool saveImageFiles, ResultLineItem &result)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();

    detectKeypoints(detector, "SIFT", keypoints, img, displayImageWindows, saveImageFiles, result);
}

void detectKeypoints(cv::Ptr<cv::FeatureDetector> &detector, const std::string& detectorName, vector<cv::KeyPoint> &keypoints, const cv::Mat &img, bool displayImageWindows, bool saveImageFiles, ResultLineItem &result)
{
    auto t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    double duration = 1000 * t / 1.0;
    cout << detectorName << " with n= " << keypoints.size() << " keypoints in " << duration << " ms" << endl;

    result.keypointCount.totalKeypoints = keypoints.size();
    result.keypointCount.descriptorMatchingTime = duration;

    string windowName = detectorName + "_Keypoint_Detection_Results";
    visualizeKeypoints(keypoints, img, windowName, displayImageWindows, saveImageFiles, result);
}

void visualizeKeypoints(const vector<cv::KeyPoint> &keypoints, const cv::Mat &img, const string& windowName, bool displayImageWindows, bool saveImageFiles, ResultLineItem &result)
{
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img,
                      keypoints,
                      visImage,
                      cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    if (displayImageWindows)
    {
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    if(saveImageFiles)
    {
        string dir_keypoint_detections = "../results/images/keypoint_detections/";
        string imageFileNameDetections = dir_keypoint_detections + windowName + ".png";
        try
        {
            cv::imwrite(imageFileNameDetections, visImage);
        }
        catch (const std::exception &ex)
        {
            std::cerr << "Image write exception: " << ex.what() << std::endl;
        }
    }

}


