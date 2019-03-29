#include "MYSVM.h"
#include<sys/time.h>
#include<unistd.h>
using namespace std;
using namespace cv;
using namespace cv::ml;
MYSVM::MYSVM()
{
    svm = SVM::create(); 
}
MYSVM::MYSVM(int _rows,int _cols)
{
    rows=_rows;
    cols=_cols;
    svm = SVM::create();
}
void MYSVM::hog(vector<float>&hog,Mat&src_img)
{
    if(src_img.rows!=rows||src_img.cols!=cols)
    {
        if(src_img.rows>0&&src_img.cols>0)
        {
            resize(src_img,src_img,Size(cols,rows));
        }     
    }
    gettimeofday(&hogStart,NULL);
    hog.clear();
#ifdef DEBUG
    ofstream f("test.txt");
#endif 

    //resize(src_img,src_img,Size(src_img.cols/8*8,src_img.rows/8*8));
#ifdef DEBUG
    cout<<src_img.cols<<" "<<src_img.rows<<endl;
#endif
#ifdef DEBUG
    imshow("test.jpg",src_img);
    waitKey();
#endif
    //convert from RGB to GRAY;
    Mat src_img_gray(Size(src_img.cols,src_img.rows),CV_8UC1,Scalar(0));
    for(int i=0;i<src_img.rows;i++)
    {
        for(int j=0;j<src_img.cols;j++)
        {
            src_img_gray.at<uchar>(i,j)=(int)(src_img.at<Vec3b>(i,j)[0]*0.3+src_img.at<Vec3b>(i,j)[1]*0.59+src_img.at<Vec3b>(i,j)[2]*0.11);
        }
    }
    //end
#ifdef DEBUG
    imshow("test.jpg",src_img_gray);
    waitKey();
#endif
    //Gamma 
    for(int i=0;i<src_img_gray.rows;i++)
    {
        for(int j=0;j<src_img_gray.cols;j++)
        {
            float gamma=src_img_gray.at<uchar>(i,j)/255.0;
            //cout<<"before"<<gamma<<endl;
            gamma=sqrt(gamma);
            //cout<<"after"<<gamma<<"  "<<(int)(gamma*255.0)<<endl;
            src_img_gray.at<uchar>(i,j)=(int)(gamma*255.0);
            
        }
    }
    //Gamma end
#ifdef DEBUG
    imshow("test.jpg",src_img_gray);
    waitKey();
#endif
    //梯度计算
    Mat theta(src_img.cols,src_img.rows,CV_32F);
    for(int i=1;i<src_img_gray.rows-1;i++)
    {
        for(int j=1;j<src_img_gray.cols-1;j++)
        {
            int Gx=src_img_gray.at<uchar>(i+1,j)-src_img_gray.at<uchar>(i-1,j);
            int Gy=src_img_gray.at<uchar>(i,j+1)-src_img_gray.at<uchar>(i,j-1);
            int Gxy=(int)(sqrt(Gx*Gx+Gy*Gy));
            src_img_gray.at<uchar>(i,j)=Gxy;
            if(Gy==0&&Gx==0)
            {
                theta.at<float>(i,j)=0;
            }
            else
            {
                theta.at<float>(i,j)=float(atan2(Gy, Gx) * 180 / CV_PI);
            }
            if(theta.at<float>(i,j)<0)
            {
                theta.at<float>(i,j)+=360;
            }
#ifdef DEBUG
            cout<<theta.at<float>(i,j)<<endl;
#endif
        }
    }
    //end
#ifdef DEBUG
    imshow("test.jpg",src_img_gray);
    waitKey();
#endif
    //cell 直方图

    float store[250][250][9];
    for(int i=0;i<250;i++)
    {
        for(int j=0;j<250;j++)
        {
            for(int p=0;p<9;p++)
            {
                store[i][j][p]=0;
            }
        }
    }
    for(int i=0;i<src_img_gray.rows/8;i++)
    {
        for(int j=0;j<src_img_gray.cols/8;j++)
        {
            //every cell
            for(int p=0;p<8;p++)
            {
                for(int q=0;q<8;q++)
                {
                    int index=(int)(theta.at<float>(i*8+p,j*8+q)/40.0);
#ifdef DEBUG
                    cout<<"theta"<<theta.at<float>(i*8+p,j*8+q)<<"index"<<index<<endl;
#endif
                    if(index<0||index>=9)
                    {
                        store[i][j][0]+=src_img_gray.at<uchar>(i*8+p,j*8+q);
                    }
                    else
                    {
                        store[i][j][index]+=src_img_gray.at<uchar>(i*8+p,j*8+q);
                    }       
#ifdef PRINT_DEBUG
                    for(int index=0;index<9;index++)
                    {
                        cout<<store[i][j][index]<<endl;
                    }
#endif
                }
            }
            //归一化
            float standard;
            for(int index =0;index<9;index++)
            {
                standard+=store[i][j][index]*store[i][j][index];
            }
            standard=sqrt(standard+0.001);
            for(int index=0;index<9;index++)
            {
                store[i][j][index]=store[i][j][index]/standard;
#ifdef DEBUG
                f<<store[i][j][index]<<endl;
#endif
            }
#ifdef PRINT_DEBUG
            for(int index=0;index<9;index++)
            {
                cout<<store[i][j][index]<<endl;
            }
#endif
        }
    }
    //end

    //生成 hog向量(描述子)
    for(int i=0;i<src_img_gray.rows/8-1;i++)
    {
        for(int j=0;j<src_img_gray.cols/8-1;j++)
        {
            for(int index=0;index<9;index++)
            {
                hog.push_back(store[i][j][index]);
                hog.push_back(store[i+1][j][index]);
                hog.push_back(store[i][j+1][index]);
                hog.push_back(store[i+1][j+1][index]);
            }
        }
    }
    gettimeofday(&hogEnd,NULL);
}
void MYSVM::readFileName(std::string relative_path, std::vector<std::string> & names)
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
void MYSVM::svmTrain(std::string trueFilePath,std::string falseFilePath,int input_rows=80,int input_cols=80)
{
    rows=input_rows;
    cols=input_cols;
    vector<string> trueNames,falseNames;
    readFileName(trueFilePath,trueNames);
    readFileName(falseFilePath,falseNames);

    string filepath=trueFilePath+trueNames[0];
    Mat test=imread(filepath);
    vector<float>hogData;
    hog(hogData,test);
    Mat trainingData(trueNames.size()+falseNames.size(),hogData.size(),CV_32F);
    Mat labelsData(trueNames.size()+falseNames.size(),1,CV_32SC1);
    Mat srcImg;
    for(int i=0;i<trueNames.size();i++)
    {
        string filepath=trueFilePath+trueNames[i];
        srcImg=imread(filepath);
        vector<float>hogData;
        hog(hogData,srcImg);
        for(int j=0;j<hogData.size();j++)
        {
            trainingData.at<float>(i,j)=hogData[j];
        }
        labelsData.at<int>(i,0)=1;
    }
    for(int i=0;i<falseNames.size();i++)
    {
        string filepath=falseFilePath+falseNames[i];
        srcImg=imread(filepath);
        vector<float>hogData;
        hog(hogData,srcImg);
        
        for(int j=0;j<hogData.size();j++)
        {
            trainingData.at<float>(i+trueNames.size(),j)=hogData[j];
        }
        labelsData.at<int>(i+trueNames.size(),0)=-1;
    }
    svm->setKernel(SVM::INTER);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingData, ROW_SAMPLE, labelsData);
}
void MYSVM::loadModel(string filename)
{
    svm=StatModel::load<SVM>(filename);
}
void MYSVM::saveSvmXml(string filename)
{
    svm->save(filename);
}
float MYSVM::svmPredict(Mat&srcImg)
{
    timeval start;
    timeval end;
    vector<float>hogData; 
    hog(hogData,srcImg);
    float* store=new float[hogData.size()];
    for(int j=0;j<hogData.size();j++)
    {
        store[j]=hogData[j];
    }
    Mat predict(1,hogData.size(),CV_32F,store);
    delete store;
    float response=svm->predict(predict);
    return response;
}
void MYSVM::svmPredict(std::string filepath)
{
    gettimeofday(&predictStart,NULL);
    vector<string>names;
    readFileName(filepath,names);
    for(int i=0;i<names.size();i++)
    {
        string filename=filepath+names[i];
        Mat srcImg=imread(filename);
        float response=svmPredict(srcImg);
        cout<<names[i]<<" "<<response<<endl;
    }
    gettimeofday(&predictEnd,NULL);
}

void MYSVM::svmPredictVideo(std::string filename,int stepdivide,int mode)
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
    Mat cutFrame;
    while(videoIsOpened)
    {
        cap>>frame;
        showFrame=frame.clone();
        if(frame.empty())
            break;    
        /**
         * hog mutidetector
         * 
         */
        for(int _mode=0;_mode<3;_mode++)
        {
            int rows_temp;
            int cols_temp;
            if(_mode==1)
            {
                rows_temp=rows/2;
                cols_temp=cols/2;
            }
            if(_mode==2)
            {
                rows_temp=rows/4;
                cols_temp=cols/4;
            }

            int step_rows_length=rows_temp/stepdivide;
            int step_cols_length=cols_temp/stepdivide;
            int rowsteps=(frame.rows-rows_temp)/step_rows_length;
            int colsteps=(frame.cols-cols_temp)/step_cols_length;
            cout<<rows_temp<<" "<<cols_temp<<endl;
            cout<<step_rows_length<<" "<<step_cols_length<<" "<<rowsteps<<" "<<colsteps<<endl;
            cout<<rowsteps<<" "<<colsteps<<endl;
        
            for(int i=0;i<rowsteps;i++)
            {
                for(int j=0;j<colsteps;j++)
                {
                    cutFrame=frame(Rect(j*step_cols_length,i*step_rows_length,rows_temp,cols_temp));
                    if(svmPredict(cutFrame)>0)
                    {
                        rectangle(showFrame,Point(j*step_cols_length,i*step_rows_length),Point(j*step_cols_length+cols_temp,i*step_rows_length+rows_temp),Scalar(255,255,255));
                        //imwrite("./detect/"+to_string(rand())+to_string(rand())+to_string(i)+" "+to_string(j)+".jpg",cutFrame);
                        imshow("frame",showFrame);
                        waitKey(10);
                    } 
                }
            }
            imshow("frame",showFrame);
            waitKey();
        }
        
    }
    cap.release();
}
void MYSVM::svmPredictPic(string filePath,int stepdivide)
{
    vector<string>names;
    readFileName(filePath,names);
    for(int i=0;i<names.size();i++)
    {
        Mat cutFrame(rows,cols,CV_8UC3,Scalar(255,255,255));
        Mat frame=imread(filePath+names[i]);
        Mat showFrame=frame.clone();
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
                float test=svmPredict(cutFrame);
                if(test>0)
                {
                    rectangle(showFrame,Point(j*step_cols_length,i*step_rows_length),Point(j*step_cols_length+cols,i*step_rows_length+rows),Scalar(255,255,255));
                    //imwrite("./new/"+to_string(rand())+to_string(rand())+to_string(i)+" "+to_string(j)+".jpg",cutFrame);
                    imshow("frame",showFrame);
                    waitKey();
                } 
            }
        }
        imshow("frame",showFrame);
        waitKey();
    }
}
void MYSVM::getUsedTime(int type)
{
    if(type==0)
    {
        cout<<(1000000*(predictEnd.tv_sec-predictStart.tv_sec)+predictEnd.tv_usec-predictStart.tv_usec)/1000<<endl;
    }
    else if(type==1)
    {
        cout<<(1000000*(hogEnd.tv_sec-hogStart.tv_sec)+hogEnd.tv_usec-hogStart.tv_usec)/1000<<endl;
    }
}