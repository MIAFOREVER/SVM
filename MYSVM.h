#pragma once
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/opencv.hpp"
#include <dirent.h>
class MYSVM
{
#define PREDICT_TIME 0
#define HOG_TIME 1
public:
    void svmTrain(std::string trueFilepath,std::string falseFilepath,int rows,int cols);
    void readFileName(std::string relative_path, std::vector<std::string> & names);
    void svmPredict(std::string filepath);
    float svmPredict(cv::Mat&srcImg);
    void loadModel(std::string filename);
    void saveSvmXml(std::string filename);
    void getUsedTime(int typeNumber);
    void svmPredictVideo(std::string filename,int stepdivide,int mode);
    void svmPredictPic(std::string filePath,int);
    void makeTrainSet(std::string filePath);
private:
    cv::Ptr<cv::ml::SVM> svm;
    int rows;
    int cols;
    timeval predictStart;
    timeval predictEnd;
    timeval hogStart;
    timeval hogEnd;
    void hog(std::vector<float>& hogData,cv::Mat& srcImg);
};
