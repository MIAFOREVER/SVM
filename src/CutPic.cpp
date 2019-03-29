#include"CutPic.h"

#include<string>
#include<iostream>
#include<vector>

#include <fstream>
#include <dirent.h>

using namespace std;
using namespace cv;

void CutPic::cutPic(Mat& src,string filename)
{
    Rect rect1(0,0,src.cols/2,src.rows/2);
    Rect rect2(0,src.cols/2,src.cols/2,src.rows/2);
    Rect rect3(src.rows/2,0,src.cols/2,src.rows/2);
    Rect rect4(src.cols/2,src.rows/2,src.cols/2,src.rows/2);
    Mat src1(src,rect1);
    Mat src2(src,rect2);
    Mat src3(src,rect3);
    Mat src4(src,rect4);
    filename=filename.substr(0,filename.find(".png"));
    imwrite(filename+"_1.png",src1);
    imwrite(filename+"_2.png",src2);
    imwrite(filename+"_3.png",src3);
    imwrite(filename+"_4.png",src4);
}
void CutPic::readFileName(std::string relative_path, std::vector<std::string> & names)
{
    names.clear();
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(relative_path.c_str());
    std::cout << "file list: " << std::endl;
    while((ptr=readdir(dir))!=NULL)
    {
        /// jump '.' or '..'
        if(ptr->d_name[0] == '.'){
            continue;
        }
        std::cout << "[" << ptr->d_name << "] has been detected. " << std::endl;
        names.push_back(ptr->d_name);
    }
    closedir(dir);
}
void CutPic::createPicFile(std::string src,std::string dst)
{
    vector<string> names;
    readFileName(src,names);
    for(int i=0;i<names.size();i++)
    {
        Mat srcImg=imread(src+names[i]);
        cutPic(srcImg,dst+names[i]);
    }
}
/*
@param filename the video path
@param rows the rows you want to cut
@param cols the cols you want to cut
@param stepdivide 
@param mode
*/
void CutPic::cutVideo(std::string filename,int rows,int cols,int stepdivide,int mode)
{
    timeval start;
    timeval end;
    
    VideoCapture cap;
    bool videoIsOpened=0;
    if(mode == 0)
        cap.open(filename);
    else if(mode==1)
        cap.open(0);
    if(cap.isOpened())
    {
        cout<<"Videos have opened successful!"<<endl;
        videoIsOpened=true;
    }
    Mat frame,showFrame;

    Mat cutFrame(rows,cols,CV_8UC3,Scalar(255,255,255));

    while(videoIsOpened)
    {
        cap>>frame;
        showFrame=frame.clone();
        if(frame.empty())
        break;
        int step_rows_length=rows/stepdivide;
        int step_cols_length=cols/stepdivide;
        int rowsteps=(frame.rows-rows)/step_rows_length;
        int colsteps=(frame.cols-cols)/step_cols_length;
        
        for(int i=0;i<=rowsteps;i++)
        {
            for(int j=0;j<=colsteps;j++)
            {
                for(int p=0;p<rows;p++)
                {
                    for(int q=0;q<cols;q++)
                    {
                        cutFrame.at<Vec3b>(p,q)[0]=frame.at<Vec3b>(i*step_rows_length+p,j*step_cols_length+q)[0];
                        cutFrame.at<Vec3b>(p,q)[1]=frame.at<Vec3b>(i*step_rows_length+p,j*step_cols_length+q)[1];
                        cutFrame.at<Vec3b>(p,q)[2]=frame.at<Vec3b>(i*step_rows_length+p,j*step_cols_length+q)[2];
                        
                    }
                } 
                
                if(1)
                {
                    rectangle(showFrame,Point(j*step_cols_length,i*step_rows_length),Point(j*step_cols_length+cols,i*step_rows_length+rows),Scalar(255,255,255));
                    imwrite("../cutvideo/"+to_string(rand())+to_string(rand())+to_string(i)+" "+to_string(j)+".jpg",cutFrame);
                    imshow("frame",showFrame);
                    waitKey(10);
                } 
                
            }
        }
        imshow("frame",showFrame);
        waitKey(10);
    }
    cap.release();
}