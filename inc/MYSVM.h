/********************************************************
 * SVM selector                                         *
 *                                                      *
 * @file    MYSVM.h                                     *
 * @brief   SVM selector                                *
 * @author  ZhangHao                                    *
 * @email   zhangh527@mail2.sysu.edu                    *
 * @date    2019/03/08                                  *
 *                                                      *
 * -----------------------------------------------------*
 * Remark   : Description                               *
 * -----------------------------------------------------*
 * Change History :                                     *
 * <Date>   |<Version>  |<Author>   |<Description>      *
 * 2019/03/08|0.1.0     |ZhangHao   |                   *
 * 2019/03/08|0.1.1     |ZhangHao   |update new function*
 *                                   and clean the code *
 * *****************************************************/
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
/**
 * @brief SVM predictor
 * 
 */
class MYSVM
{
#define PREDICT_TIME 0
#define HOG_TIME 1
public:
    int rows;
    int cols;
    /**
     * @brief Construct a new MYSVM object
     * 
     */
    MYSVM();
    /**
     * @brief Construct a new MYSVM object
     * 
     * @param rows      Init rows 
     * @param cols      Init cols
     */
    MYSVM(int rows,int cols);
    /**
     * @brief           A simple interface to use SVM
     * You can store the picture you want in trueFilePath and store the picture you don't want in falseFilePath
     * @param trueFilepath  
     * @param falseFilepath 
     * @param rows 
     * @param cols 
     */
    void svmTrain(std::string trueFilepath,std::string falseFilepath,int rows,int cols);
    /**
     * @brief           If you have run svmTrain(or load SVM model). You can use svmPredict function to predict a floder. Output format is filename+1(true)/0(false)
     * 
     * @param filepath  filepath
     */
    void svmPredict(std::string filepath);
    /**
     * @brief           A user interface to predict one picture
     * 
     * @param srcImg 
     * @return float    1 is true 0 is false
     */
    float svmPredict(cv::Mat&srcImg);
    /**
     * @brief           Load SVM model from user path(Must end with *.xml)
     * 
     * @param filename 
     */
    void loadModel(std::string filename);
    /**
     * @brief           Save SVM model from user path(Must end with *.xml)
     * 
     * @param filename 
     */
    void saveSvmXml(std::string filename);
    void getUsedTime(int typeNumber);
    /**
     * @brief           Select the true aera in your video
     * 
     * @param filename 
     * @param stepdivide 
     * @param mode      1 is computer camera and 0 is intput videofile
     */
    void svmPredictVideo(std::string filename,int stepdivide,int mode=0);
    /**
     * @brief           A brief application in picture
     *                  update the mutidetector
     * @param filePath 
     * @param mode      1 is computer camera and 0 is intput videofile
     */
    void svmPredictPic(std::string filePath,int mode);
private:
    void readFileName(std::string relative_path, std::vector<std::string> & names);
    void hog(std::vector<float>& hogData,cv::Mat& srcImg);
    cv::Ptr<cv::ml::SVM> svm;
    
    timeval predictStart;
    timeval predictEnd;
    timeval hogStart;
    timeval hogEnd;
};
