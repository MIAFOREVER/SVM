/****************************************************
 * Cuting picture from origin to 4-divided pictures *
 *                                                  *
 * @file    CutPic.h                                *
 * @brief   cut picture                             *
 * @author  ZhangHao                                *
 * @email   zhangh527@mail2.sysu.edu                *
 * @date    2019/03/08                              *
 *                                                  *
 * -------------------------------------------------*
 * Remark   : Description                           *
 * -------------------------------------------------*
 * Change History :                                 *
 * <Date>   |<Version>  |<Author>   |<Description>  *
 * 2019/03/08|0.1.0     |ZhangHao   |               *
 *                                                  *
 * *************************************************/
#pragma once
#include<opencv2/opencv.hpp>
/**
 * @brief Cuting picture
 * A simple tools to cut pictures
 */
class CutPic
{
    public:
    /**
     * @brief Create a picture
     * 
     * @param src       Input filepath
     * @param dst       Output filepath
     */
    void createPicFile(std::string src,std::string dst);
    /**
     * @brief               A function used as cut negetive 
     * 
     * @param filename      Intput filepath
     * @param rows          Input Rows
     * @param cols          Input Cols
     * @param stepdivide    
     * @param mode          1 is computer camera and 0 is filename
     */
    void cutVideo(std::string filename,int rows,int cols,int stepdivide,int mode=0);
    private:
    void readFileName(std::string relative_path, std::vector<std::string> & names);
    /**
     * @brief Init function
     * 
     * @param src       Input Mat 
     * @param filePath  Output filepath
     */
    void cutPic(cv::Mat& src,std::string filePath);
};