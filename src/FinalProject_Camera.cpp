
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <boost/circular_buffer.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

void DisplayResultsTable(const ResultSet & results);

void DisplayImagesTable(const ResultSet & results);

vector<ResultSet> RunExperimentSet(Hyperparameters hyperparameters,
                                   const std::vector<string> & detectors,
                                   const std::vector<string> & descriptors);

void RunExperiment(Experiment & experiment, ResultSet & results);

int main()
{
    Hyperparameters hyperparameters = Hyperparameters();

    std::vector<string> detectors = {
            "Shi_Tomasi",
//            "HARRIS",
//            FAST,
//            BRISK,
//            ORB,
//            AKAZE,
            "SIFT"
    };
    std::vector<string> descriptors = {
            "BRISK"//,
//            "BRIEF",
//            "ORB",
//            "FREAK",
//            "AKAZE",
//            "SIFT"
    };

    std::vector<ResultSet> resultSet = RunExperimentSet(hyperparameters, detectors, descriptors);

    for(const auto & results : resultSet)
    {
        DisplayResultsTable(results);
        DisplayImagesTable(results);
    }

    return 0;
}

vector<ResultSet> RunExperimentSet(Hyperparameters hyperparameters,
                                   const std::vector<string> & detectors,
                                   const std::vector<string> & descriptors)
{
    unsigned int experimentCount = 0;
    std::vector<ResultSet> resultSet;

    for(const auto & detector:detectors)
    {
        for(const auto & descriptor:descriptors)
        {
            cout << "\n*** RUNNING EXPERIMENT " << experimentCount << " WITH detector = " << detector
                 << "  and descriptor = " << descriptor << " ***" << endl;
            ResultSet results;

            hyperparameters.descriptor = descriptor;

            Experiment experiment = Experiment();
            experiment.hyperparameters = hyperparameters;
            experiment.hyperparameters.keypointDetector = detector;
            experiment.hyperparameters.descriptor = descriptor;

            // There are some combinations of detector and descriptor that do not work together:
            if(descriptor == "AKAZE")
            {
                // AKAZE descriptors work only with AKAZE detectors.
                if(detector == "AKAZE")
                {
                    RunExperiment(experiment, results);
                }
                else
                {
                    cerr << "Can't use AKAZE descriptor with non-AKAZE detector." << endl;
                    continue;
                }
            }

            if(descriptor == "ORB")
            {
                // ORB detectors do not work with SIFT descriptors.
                if(detector != "SIFT")
                {
                    RunExperiment(experiment, results);
                }
                else
                {
                    cerr << "Can't use ORB descriptor with SIFT detector." << endl;
                    continue;
                }
            }

            // All other detector-descriptor combinations are assumed to be valid
            RunExperiment(experiment, results);

            resultSet.push_back(results);

            experimentCount++;
        }
    }

    cout << "# Performance Evaluation" << endl;
    cout << "These results are recorded from running a total of " << experimentCount
         << " experiments based on combinations of " << detectors.size() << " detectors and " << descriptors.size()
         << " descriptors." << endl;

    return resultSet;
}


/*
 * This function encapsulates running an experiment with a given combination of detector, descriptor, matcher,
 * descriptor type, and selector.
 */
void RunExperiment(Experiment & experiment, ResultSet & results)
{
    unsigned int firstImage = 0;
    unsigned int secondImage = 1;

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1;   // increase or decrease the frame rate, e.g., 2=skip every second frame; 1=use every frame
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3, 4, cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4, 4, cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4, 4, cv::DataType<double>::type); // rotation matrix and translation vector

    RT.at<double>(0, 0) = 7.533745e-03;
    RT.at<double>(0, 1) = -9.999714e-01;
    RT.at<double>(0, 2) = -6.166020e-04;
    RT.at<double>(0, 3) = -4.069766e-03;
    RT.at<double>(1, 0) = 1.480249e-02;
    RT.at<double>(1, 1) = 7.280733e-04;
    RT.at<double>(1, 2) = -9.998902e-01;
    RT.at<double>(1, 3) = -7.631618e-02;
    RT.at<double>(2, 0) = 9.998621e-01;
    RT.at<double>(2, 1) = 7.523790e-03;
    RT.at<double>(2, 2) = 1.480755e-02;
    RT.at<double>(2, 3) = -2.717806e-01;
    RT.at<double>(3, 0) = 0.0;
    RT.at<double>(3, 1) = 0.0;
    RT.at<double>(3, 2) = 0.0;
    RT.at<double>(3, 3) = 1.0;

    R_rect_00.at<double>(0, 0) = 9.999239e-01;
    R_rect_00.at<double>(0, 1) = 9.837760e-03;
    R_rect_00.at<double>(0, 2) = -7.445048e-03;
    R_rect_00.at<double>(0, 3) = 0.0;
    R_rect_00.at<double>(1, 0) = -9.869795e-03;
    R_rect_00.at<double>(1, 1) = 9.999421e-01;
    R_rect_00.at<double>(1, 2) = -4.278459e-03;
    R_rect_00.at<double>(1, 3) = 0.0;
    R_rect_00.at<double>(2, 0) = 7.402527e-03;
    R_rect_00.at<double>(2, 1) = 4.351614e-03;
    R_rect_00.at<double>(2, 2) = 9.999631e-01;
    R_rect_00.at<double>(2, 3) = 0.0;
    R_rect_00.at<double>(3, 0) = 0;
    R_rect_00.at<double>(3, 1) = 0;
    R_rect_00.at<double>(3, 2) = 0;
    R_rect_00.at<double>(3, 3) = 1;

    P_rect_00.at<double>(0, 0) = 7.215377e+02;
    P_rect_00.at<double>(0, 1) = 0.000000e+00;
    P_rect_00.at<double>(0, 2) = 6.095593e+02;
    P_rect_00.at<double>(0, 3) = 0.000000e+00;
    P_rect_00.at<double>(1, 0) = 0.000000e+00;
    P_rect_00.at<double>(1, 1) = 7.215377e+02;
    P_rect_00.at<double>(1, 2) = 1.728540e+02;
    P_rect_00.at<double>(1, 3) = 0.000000e+00;
    P_rect_00.at<double>(2, 0) = 0.000000e+00;
    P_rect_00.at<double>(2, 1) = 0.000000e+00;
    P_rect_00.at<double>(2, 2) = 1.000000e+00;
    P_rect_00.at<double>(2, 3) = 0.000000e+00;

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    boost::circular_buffer<DataFrame> dataBuffer(dataBufferSize); // buffer of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results


    // Track the results
    ResultLineItem result;
    const bool visualizeImages = experiment.displayImageWindows;

    /* MAIN LOOP OVER ALL IMAGES */

    for(size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex += imgStepWidth)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file 
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;
        detectObjects((dataBuffer.end() - 1)->cameraImg,
                      (dataBuffer.end() - 1)->boundingBoxes,
                      confThreshold,
                      nmsThreshold,
                      yoloBasePath,
                      yoloClassesFile,
                      yoloModelConfiguration,
                      yoloModelWeights,
                      bVis);

        cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        // NOTE: this implementation assumes a level road surface; a steep incline, e.g., going up a hill, would cause a problem
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

        cout << "#3 : CROP LIDAR POINTS done" << endl;


        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI((dataBuffer.end() - 1)->boundingBoxes,
                            (dataBuffer.end() - 1)->lidarPoints,
                            shrinkFactor,
                            P_rect_00,
                            R_rect_00,
                            RT);

        // Visualize 3D objects
        bVis = true;
        if(bVis)
        {
            show3DObjects((dataBuffer.end() - 1)->boundingBoxes,
                          cv::Size(4.0, 20.0),
                          cv::Size(2000, 2000),
                          false,
                          result,
                          experiment.hyperparameters.keypointDetector,
                          experiment.hyperparameters.descriptor);
        }
        bVis = false;

        cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;


        // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
        //continue; // skips directly to the next image without processing what comes beneath

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end() - 1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
//        string detectorType = "SHITOMASI";
//        results.detector = detectorType;

        result.keypointMatch.matchedImagePair.first = firstImage;
        result.keypointMatch.matchedImagePair.second = secondImage;

        if(experiment.hyperparameters.keypointDetector == "SIFT" || experiment.hyperparameters.descriptor == "SIFT")
        {
            experiment.hyperparameters.matcherType = "MAT_FLANN";
            experiment.hyperparameters.descriptorType = "DES_HOG";
        }
        else
        {
            experiment.hyperparameters.matcherType = "MAT_BF";
            experiment.hyperparameters.descriptorType = "DES_BINARY";
        }


        results.detector = experiment.hyperparameters.keypointDetector;
        bool saveImagesToFile = experiment.saveKeypointDetectionImagesToFile;

        if(experiment.hyperparameters.keypointDetector == "Shi_Tomasi")
        {
            detKeypointsShiTomasi(keypoints, imgGray, visualizeImages, saveImagesToFile, result);
        }
        else if(experiment.hyperparameters.keypointDetector == "HARRIS")
        {
            detKeypointsHarris(keypoints, imgGray, visualizeImages, saveImagesToFile, result);
        }
        else if(experiment.hyperparameters.keypointDetector == "FAST")
        {
            detKeypointsFAST(keypoints, imgGray, visualizeImages, saveImagesToFile, result);
        }
        else if(experiment.hyperparameters.keypointDetector == "BRISK")
        {
            detKeypointsBRISK(keypoints, imgGray, visualizeImages, saveImagesToFile, result);
        }
        else if(experiment.hyperparameters.keypointDetector == "ORB")
        {
            detKeypointsORB(keypoints, imgGray, visualizeImages, saveImagesToFile, result);
        }
        else if(experiment.hyperparameters.keypointDetector == "AKAZE")
        {
            detKeypointsAKAZE(keypoints, imgGray, visualizeImages, saveImagesToFile, result);
        }
        else if(experiment.hyperparameters.keypointDetector == "SIFT")
        {
            detKeypointsSIFT(keypoints, imgGray, visualizeImages, saveImagesToFile, result);
        }
        else
        {
            cerr << "Attempting to use an unsupported keypoint detector" << endl;
        }
        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if(bLimitKpts)
        {
            int maxKeypoints = 50;

            if(experiment.hyperparameters.keypointDetector == "Shi_Tomasi")
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        cout << "#5 : DETECT KEYPOINTS done" << endl;


        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        string descriptorType = experiment.hyperparameters.descriptor; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
        descKeypoints((dataBuffer.end() - 1)->keypoints,
                      (dataBuffer.end() - 1)->cameraImg,
                      descriptors,
                      descriptorType,
                      result);


        results.descriptor = descriptorType;

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


        if(dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;


            string matcherType = experiment.hyperparameters.matcherType;        // MAT_BF, MAT_FLANN
            string descriptorType = experiment.hyperparameters.descriptorType; // DES_BINARY, DES_HOG
            string selectorType = experiment.hyperparameters.selectorType;       // SEL_NN, SEL_KNN

            try
            {
                result.keypointMatch.matchedImagePair.first = firstImage;
                result.keypointMatch.matchedImagePair.second = secondImage;

                if(experiment.hyperparameters.keypointDetector == "SIFT" ||
                   experiment.hyperparameters.descriptor == "SIFT")
                {
                    experiment.hyperparameters.matcherType = "MAT_FLANN";
                    experiment.hyperparameters.descriptorType = "DES_HOG";
                }
                else
                {
                    experiment.hyperparameters.matcherType = "MAT_BF";
                    experiment.hyperparameters.descriptorType = "DES_BINARY";
                }
                matchDescriptors((dataBuffer.end() - 2)->keypoints,
                                 (dataBuffer.end() - 1)->keypoints,
                                 (dataBuffer.end() - 2)->descriptors,
                                 (dataBuffer.end() - 1)->descriptors,
                                 matches,
                                 descriptorType,
                                 matcherType,
                                 selectorType,
                                 result);
            }
            catch(const std::exception & ex)
            {
                std::cerr << "Exception calling matchDescriptors(): " << ex.what() << std::endl;
            }

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;


            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            map<int, int> boundingBoxBestMatches;
            DataFrame & previousFrame = *(dataBuffer.end() - 2);
            DataFrame & currentFrame = *(dataBuffer.end() - 1);

            // associate bounding boxes between current and previous frame using keypoint matches
            matchBoundingBoxes(matches,
                               boundingBoxBestMatches,
                               previousFrame,
                               currentFrame);
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->bbMatches = boundingBoxBestMatches;

            cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


            /* COMPUTE TTC ON OBJECT IN FRONT */

            // loop over all BB match pairs
            for(auto it1 = (dataBuffer.end() - 1)->bbMatches.begin();
                it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
            {
                // find bounding boxes associates with current match
                BoundingBox *boundingBoxPreviousFrame, *boundingBoxCurrentFrame;
                for(auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin();
                    it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                {
                    if(it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        boundingBoxCurrentFrame = &(*it2);
                    }
                }

                for(auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin();
                    it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                {
                    if(it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        boundingBoxPreviousFrame = &(*it2);
                    }
                }

                // compute TTC for current match
                if(boundingBoxCurrentFrame->lidarPoints.size() > 0 &&
                   boundingBoxPreviousFrame->lidarPoints.size() > 0) // only compute TTC if we have Lidar points
                {
                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                    double ttcLidar;
                    computeTTCLidar(boundingBoxPreviousFrame->lidarPoints,
                                    boundingBoxCurrentFrame->lidarPoints,
                                    sensorFrameRate,
                                    ttcLidar);
                    //// EOF STUDENT ASSIGNMENT

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                    double ttcCamera;
                    vector<cv::KeyPoint> & keypointsPreviousFrame = (dataBuffer.end() - 2)->keypoints;
                    vector<cv::KeyPoint> & keypointsCurrentFrame = (dataBuffer.end() - 1)->keypoints;
                    vector<cv::DMatch> & keypointMatches = (dataBuffer.end() - 1)->kptMatches;

                    clusterKptMatchesWithROI(*boundingBoxCurrentFrame,
                                             keypointsPreviousFrame,
                                             keypointsCurrentFrame,
                                             keypointMatches);

                    computeTTCCamera(keypointsPreviousFrame,
                                     keypointsCurrentFrame,
                                     boundingBoxCurrentFrame->kptMatches,
                                     sensorFrameRate,
                                     ttcCamera);
                    //// EOF STUDENT ASSIGNMENT

                    bVis = true;
                    if(bVis)
                    {
                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        showLidarImgOverlay(visImg,
                                            boundingBoxCurrentFrame->lidarPoints,
                                            P_rect_00,
                                            R_rect_00,
                                            RT,
                                            &visImg);
                        cv::rectangle(visImg,
                                      cv::Point(boundingBoxCurrentFrame->roi.x, boundingBoxCurrentFrame->roi.y),
                                      cv::Point(boundingBoxCurrentFrame->roi.x + boundingBoxCurrentFrame->roi.width,
                                                boundingBoxCurrentFrame->roi.y + boundingBoxCurrentFrame->roi.height),
                                      cv::Scalar(0, 255, 0),
                                      2);

                        char str[200];
                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255));

                        string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
//                        cv::imshow(windowName, visImg);
//                        cout << "Press key to continue to next frame" << endl;
//                        cv::waitKey(0);

                        string imageFileName = "../" + GetTtcFilename(results.detector, results.descriptor, result.frame);
                        cv::imwritemulti(imageFileName, visImg);
                    }
                    bVis = false;

                    result.frame = imgIndex;
                    result.ttcLidar = ttcLidar;
                    result.ttcCamera = ttcCamera;
                    result.lidarPoints = currentFrame.lidarPoints.size();

                } // eof TTC computation

                firstImage++;
                secondImage++;
            } // eof loop over all BB matches
            results.data.push_back(result);
        }
    } // eof loop over all images
}

void DisplayResultsTable(const ResultSet & results)
{
    const string separator = " | ";
    cout << "\nPerformance Results" << endl;
    cout << "* Detector = " << results.detector << endl;
    cout << "* Descriptor = " << results.descriptor << endl << endl;

    cout << "Frame" << separator << "Lidar points" << separator << "TTC Lidar" << separator << "TTC Camera" << endl;
    cout << "---: " << separator << "---: " << separator << "---: " << separator << "---: " << endl;

    for(const auto & result : results.data)
    {
        cout << result.frame << separator << result.lidarPoints << separator << result.ttcLidar << separator
             << result.ttcCamera << endl;
    }
}

void DisplayImagesTable(const ResultSet & results)
{
    string imageFileNameLidarTTC;
    string imageFileNameCameraTTC;

    const string separator = " | ";
    cout << "\nPerformance Results" << endl;
    cout << "* Detector = " << results.detector << endl;
    cout << "* Descriptor = " << results.descriptor << endl << endl;

    cout << "Frame" << separator << "Top view perspective of Lidar points showing distance markers" << separator
         << "Image with TTC estimates from Lidar and Camera" << separator << "Lidar points" << separator << "TTC Lidar"
         << separator << "TTC Camera" << endl;
    cout << ":---: " << separator << ":---: " << separator << ":---: " << separator << "---: " << separator << "---: "
         << separator << "---: " << endl;

    for(const auto & result : results.data)
    {
        imageFileNameLidarTTC = GetLidarFilename(results.detector, results.descriptor, result.frame);
        imageFileNameCameraTTC = GetTtcFilename(results.detector, results.descriptor, result.frame);

        cout << result.frame << separator << "![](" << imageFileNameLidarTTC << ")" << separator << "![]("
             << imageFileNameCameraTTC << ")" << separator << result.lidarPoints << separator << result.ttcLidar
             << separator << result.ttcCamera << endl;
    }
}
