# Track an Object in 3D Space

This is the final project of the sensor fusion camera course. The following program schematic outlines what is included in this project.

![](images/course_code_structure.png)

This project implements the missing parts in the schematic. To accomplish this, there are four major tasks to complete: 
1. First, develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, compute the TTC based on Lidar measurements. 
3. Then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, conduct various tests with the framework. The goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. 

# Contents

* [Project Specification](#Project-Specification)
* [Project Report](#Project-Report)
  * [Match 3D Objects](#Match-3D-Objects)
  * [Compute Lidar-based TTC](#Compute-Lidar-based-TTC)
  * [Associate Keypoint Correspondences with Bounding Boxes](#Associate-Keypoint-Correspondences-with-Bounding-Boxes)
  * [Compute Camera-based TTC](#Compute-Camera-based-TTC)
* [Performance Evaluation](#Performance-Evaluation)
  * [Performance Evaluation 1](#Performance-Evaluation-1)
  * [Performance Evaluation 2](#Performance-Evaluation-2)
* [Building and Running the Project](#Building-and-Running-the-Project)
* [References](#References)


# Project Specification

#|Req| Criteria | Specification | Status
:--- | :--- | :--- | :--- | :---
FP.0|Report|Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.|The writeup / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled.| IN PROGRESS
FP.1|Match 3D Objects|Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.|Code is functional and returns the specified output, where each bounding box is assigned the match candidate with the highest number of occurrences.| PLANNED
FP.2|Compute Lidar-based TTC|Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.|Code is functional and returns the specified output. Also, the code is able to deal with outlier Lidar points in a statistically robust way to avoid severe estimation errors.| PLANNED
FP.3|Associate Keypoint Correspondences with Bounding Boxes|Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.|Code performs as described and adds the keypoint correspondences to the "kptMatches" property of the respective bounding boxes. Also, outlier matches have been removed based on the euclidean distance between them in relation to all the matches in the bounding box.| PLANNED
FP.4|Compute Camera-based TTC|Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.|Code is functional and returns the specified output. Also, the code is able to deal with outlier correspondences in a statistically robust way to avoid severe estimation errors.| PLANNED
FP.5|Performance Evaluation 1|Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.|Several examples (2-3) have been identified and described in detail. The assertion that the TTC is off has been based on manually estimating the distance to the rear of the preceding vehicle from a top view perspective of the Lidar points.| PLANNED
FP.6|Performance Evaluation 2|Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.|All detector / descriptor combinations implemented in previous chapters have been compared with regard to the TTC estimate on a frame-by-frame basis. To facilitate comparison, a spreadsheet and graph should be used to represent the different TTCs.| PLANNED


# Project Report

## Match 3D Objects
The goal of this task is to implement the method `matchBoundingBoxes`, which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property)“. Matches must be the ones with the highest number of keypoint correspondences.

The task is complete once the code is functional and returns the specified output, where each bounding box is assigned the match candidate with the highest number of occurrences.

The method `matchBoundingBoxes` is called from [FinalProject_Camera.cpp](src/FinalProject_Camera.cpp) like this:
```c++
map<int, int> boundingBoxBestMatches;
DataFrame & previousFrame = *(dataBuffer.end() - 2);
DataFrame & currentFrame = *(dataBuffer.end() - 1);

// associate bounding boxes between current and previous frame using keypoint matches
matchBoundingBoxes(matches,
                   boundingBoxBestMatches,
                   previousFrame,
                   currentFrame);
```

This is the implementation of the method `matchBoundingBoxes` in [camFusion_Student.cpp](src/camFusion_Student.cpp):

```c++
void matchBoundingBoxes(std::vector<cv::DMatch> & matches,
                        std::map<int, int> & boundingBoxBestMatches,
                        DataFrame & previousFrame,
                        DataFrame & currentFrame)
{
    std::multimap<int, int> boundingBoxMatches{};  // A pair of IDs to track bounding boxes

    for(auto & match : matches)
    {
        cv::KeyPoint keypointsPreviousFrame = previousFrame.keypoints[match.queryIdx];
        cv::KeyPoint keypointsCurrentFrame = currentFrame.keypoints[match.trainIdx];

        unsigned int boxIdPreviousFrame;
        unsigned int boxIdCurrentFrame;

        for(auto & boundingBox : previousFrame.boundingBoxes)
        {
            if(boundingBox.roi.contains(keypointsPreviousFrame.pt))
            {
                boxIdPreviousFrame = boundingBox.boxID;
            }
        }

        for(auto & boundingBox : currentFrame.boundingBoxes)
        {
            if(boundingBox.roi.contains(keypointsCurrentFrame.pt))
            {
                boxIdCurrentFrame = boundingBox.boxID;
            }
        }

        boundingBoxMatches.insert({boxIdCurrentFrame, boxIdPreviousFrame});
    }

    vector<int> boundingBoxIdsCurrentFrame{};

    for(auto & boundingBox : currentFrame.boundingBoxes)
    {
        boundingBoxIdsCurrentFrame.push_back(boundingBox.boxID);
    }

    for(int boxIdCurrentFrame : boundingBoxIdsCurrentFrame)
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

        for(auto & boxIdMatch : boundingBoxIdMatches)
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
```


## Compute Lidar-based TTC

The goal of this part of the project is to compute the time-to-collision for all matched 3D objects based on Lidar measurements alone. 

The estimation is implemented in a way that makes it robust against outliers which might be way too close and thus lead to faulty estimates of the TTC. The TCC is returned to the main function at the end of `computeTTCLidar`.

The task is complete once the code is functional and returns the specified output. Also, the code is able to deal with outlier Lidar points in a statistically robust way to avoid severe estimation errors.

```c++
void computeTTCLidar(std::vector<LidarPoint> & lidarPointsPreviousFrame,
                     std::vector<LidarPoint> & lidarPointsCurrentFrame,
                     double frameRate,
                     double & TTC)
{
    std::cout << "Lidar Previous Frame: " << lidarPointsPreviousFrame.size() << " points" << "\t Lidar Current Frame: "
              << lidarPointsCurrentFrame.size() << " points" << std::endl;

    // auxiliary variables
    double dT = 0.1 / frameRate;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;

    for(auto it = lidarPointsPreviousFrame.begin(); it != lidarPointsPreviousFrame.end(); ++it)
    {
        if(abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
        }
    }

    for(auto it = lidarPointsCurrentFrame.begin(); it != lidarPointsCurrentFrame.end(); ++it)
    {
        if(abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
        }
    }

    std::cout << "Final minXPrev: " << minXPrev << "\t Final minXCurr: " << minXCurr << std::endl;

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}
```

## Associate Keypoint Correspondences with Bounding Boxes

Before a TTC estimate can be computed in the next part of the project, we need to find all keypoint matches that belong to each 3D object. 

We can do this by simply checking whether the corresponding keypoints are within the region of interest in the camera image. All matches which satisfy this condition should be added to a vector. The problem you will find is that there will be outliers among the matches. To eliminate those, we compute a robust mean of all the euclidean distances between keypoint matches and then remove those that are too far away from the mean.

The task is complete once the code performs as described and adds the keypoint correspondences to the `kptMatches` property of the respective bounding boxes. Also, outlier matches have been removed based on the euclidean distance between them in relation to all the matches in the bounding box.

```c++
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
```

## Compute Camera-based TTC

Once keypoint matches have been added to the bounding boxes, the next step is to compute the TTC estimate. 

Once we have our estimate of the TTC, we return it to the `main` function at the end of `computeTTCCamera`.

The task is complete once the code is functional and returns the specified output. Also, the code must be able to deal with outlier correspondences in a statistically robust way to avoid severe estimation errors.

```c++
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
```

# Performance Evaluation

## Performance Evaluation 1

## Performance Evaluation 2


# Building and Running the Project

## Dependencies for Running Locally
* cmake >= 3.1
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.5.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

# References
