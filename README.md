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

**FP.0 Report**

Criteria | Specification | Status
:--- | :--- | :---
Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.|The writeup / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled.| IN PROGRESS

**FP.1 Match 3D Objects**

Criteria | Specification | Status
:--- | :--- | :---
Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.|Code is functional and returns the specified output, where each bounding box is assigned the match candidate with the highest number of occurrences.| PLANNED

**FP.2 Compute Lidar-based TTC**

Criteria | Specification | Status
:--- | :--- | :---
Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.|Code is functional and returns the specified output. Also, the code is able to deal with outlier Lidar points in a statistically robust way to avoid severe estimation errors.| PLANNED

**FP.3 Associate Keypoint Correspondences with Bounding Boxes**

Criteria | Specification | Status
:--- | :--- | :---
Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.|Code performs as described and adds the keypoint correspondences to the "kptMatches" property of the respective bounding boxes. Also, outlier matches have been removed based on the euclidean distance between them in relation to all the matches in the bounding box.| PLANNED

**FP.4 Compute Camera-based TTC**

Criteria | Specification | Status
:--- | :--- | :---
Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.|Code is functional and returns the specified output. Also, the code is able to deal with outlier correspondences in a statistically robust way to avoid severe estimation errors.| PLANNED

**FP.5 Performance Evaluation 1**

Criteria | Specification | Status
:--- | :--- | :---
Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.|Several examples (2-3) have been identified and described in detail. The assertion that the TTC is off has been based on manually estimating the distance to the rear of the preceding vehicle from a top view perspective of the Lidar points.| PLANNED

**FP.6 Performance Evaluation 2**

Criteria | Specification | Status
:--- | :--- | :---
Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.|All detector / descriptor combinations implemented in previous chapters have been compared with regard to the TTC estimate on a frame-by-frame basis. To facilitate comparison, a spreadsheet and graph should be used to represent the different TTCs.| PLANNED


# Project Report
Criteria | Specification | Status
:--- | :--- | :---
 |  | PLANNED 


## Match 3D Objects

## Compute Lidar-based TTC

## Associate Keypoint Correspondences with Bounding Boxes

## Compute Camera-based TTC

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
