#include "MYSVM.h"
#include<sys/time.h>
#include<unistd.h>
using namespace std;
using namespace cv;
using namespace cv::ml;

void MYSVM::hog(vector<float>&hog,Mat&src_img)
{
    //if(src_img.rows!=rows||src_img.cols!=cols)
    //{
        //cout<<"successful\n";
        //resize(src_img,src_img,Size(cols,rows));
    //}
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
    svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::INTER);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingData, ROW_SAMPLE, labelsData);
    svm->save("SVM.xml");
}
void MYSVM::loadModel(std::string filename)
{
    svm->load(filename);
}
void MYSVM::saveSvmXml(string filename)
{
    svm->save("SVM.xml");
}
float MYSVM::svmPredict(cv::Mat&srcImg)
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
        cout<<names[i]<<" "<<response<<endl;
    }
    gettimeofday(&predictEnd,NULL);
}

void MYSVM::svmPredictVideo(std::string filename,int stepdivide,int mode=0)
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
    /*
    while(1)
    {
        cap>>showFrame;
        cout<<showFrame.cols<<" "<<showFrame.rows<<endl;
        imshow("test",showFrame);
        waitKey(20);
    }
    */

    Mat cutFrame(rows,cols,CV_8UC3,Scalar(255,255,255));
    //cout<<"cutFrame: "<<cutFrame.rows<<" "<<cutFrame.cols<<endl;
    while(videoIsOpened)
    {
        cap>>frame;
        //int rows=(int)(showFrame.rows/size.width*size.width);
        //int cols=(int)(showFrame.cols/size.height*size.height);
        //Size Frame=Size(rows,cols);
        
        showFrame=frame.clone();
        //resize(frame,frame,Frame);
        if(frame.empty())
        break;
        int step_rows_length=rows/stepdivide;
        int step_cols_length=cols/stepdivide;
        int rowsteps=(frame.rows-rows)/step_rows_length;
        int colsteps=(frame.cols-cols)/step_cols_length;
        cout<<step_rows_length<<" "<<step_cols_length<<" "<<rowsteps<<" "<<colsteps<<endl;
        cout<<rowsteps<<" "<<colsteps<<endl;
        
        for(int i=0;i<=rowsteps;i++)
        {
            for(int j=0;j<=colsteps;j++)
            {
                for(int p=0;p<rows;p++)
                {
                    for(int q=0;q<cols;q++)
                    {
                        //cout<<j*step_cols_length+q<<" "<<i*step_rows_length+p<<endl;
                        cutFrame.at<Vec3b>(p,q)[0]=frame.at<Vec3b>(i*step_rows_length+p,j*step_cols_length+q)[0];
                        cutFrame.at<Vec3b>(p,q)[1]=frame.at<Vec3b>(i*step_rows_length+p,j*step_cols_length+q)[1];
                        cutFrame.at<Vec3b>(p,q)[2]=frame.at<Vec3b>(i*step_rows_length+p,j*step_cols_length+q)[2];
                        
                    }
                }
                //hog(hogData,cutFrame); 
                
                if(svmPredict(cutFrame)>0)
                {
                    
                    //line(showFrame,Point(i*step_rows_length,j*step_cols_length),Point(i*step_rows_length+size.width,j*step_cols_length),Scalar(255,255,255));
                    //line(showFrame,Point(i*step_rows_length+size.width,j*step_cols_length),Point(i*step_rows_length+size.width,j*step_cols_length+size.height),Scalar(255,255,255));
                    //line(showFrame,Point(i*step_rows_length+size.width,j*step_cols_length+size.height),Point(i*step_rows_length,j*step_cols_length+size.height),Scalar(255,255,255));
                    //line(showFrame,Point(i*step_rows_length,j*step_cols_length+size.height),Point(i*step_rows_length,j*step_cols_length),Scalar(255,255,255));
                    rectangle(showFrame,Point(j*step_cols_length,i*step_rows_length),Point(j*step_cols_length+cols,i*step_rows_length+rows),Scalar(255,255,255));
                    imwrite("./detect/"+to_string(rand())+to_string(rand())+to_string(i)+" "+to_string(j)+".jpg",cutFrame);
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
        cout<<step_rows_length<<" "<<step_cols_length<<" "<<rowsteps<<" "<<colsteps<<endl;
        cout<<rowsteps<<" "<<colsteps<<endl;
        
        for(int i=0;i<=rowsteps;i++)
        {
            for(int j=0;j<=colsteps;j++)
            {
                for(int p=0;p<rows;p++)
                {
                    for(int q=0;q<cols;q++)
                    {
                        //cout<<j*step_cols_length+q<<" "<<i*step_rows_length+p<<endl;
                        cutFrame.at<Vec3b>(p,q)[0]=frame.at<Vec3b>(i*step_rows_length+p,j*step_cols_length+q)[0];
                        cutFrame.at<Vec3b>(p,q)[1]=frame.at<Vec3b>(i*step_rows_length+p,j*step_cols_length+q)[1];
                        cutFrame.at<Vec3b>(p,q)[2]=frame.at<Vec3b>(i*step_rows_length+p,j*step_cols_length+q)[2];
                        
                    }
                }
                    //hog(hogData,cutFrame); 
                
                if(svmPredict(cutFrame)>0)
                {
                    
                    //line(showFrame,Point(i*step_rows_length,j*step_cols_length),Point(i*step_rows_length+size.width,j*step_cols_length),Scalar(255,255,255));
                    //line(showFrame,Point(i*step_rows_length+size.width,j*step_cols_length),Point(i*step_rows_length+size.width,j*step_cols_length+size.height),Scalar(255,255,255));
                    //line(showFrame,Point(i*step_rows_length+size.width,j*step_cols_length+size.height),Point(i*step_rows_length,j*step_cols_length+size.height),Scalar(255,255,255));
                    //line(showFrame,Point(i*step_rows_length,j*step_cols_length+size.height),Point(i*step_rows_length,j*step_cols_length),Scalar(255,255,255));
                    rectangle(showFrame,Point(j*step_cols_length,i*step_rows_length),Point(j*step_cols_length+cols,i*step_rows_length+rows),Scalar(255,255,255));
                    imwrite("./new/"+to_string(rand())+to_string(rand())+to_string(i)+" "+to_string(j)+".jpg",cutFrame);
                    imshow("frame",showFrame);
                    
                    waitKey(10);
                } 
                
            }
        }
        imshow("frame",showFrame);
        waitKey(1000);
    }
}
void MYSVM::makeTrainSet(std::string filePath)
{
    srand(time(NULL));
    CascadeClassifier eye_Classifier;//载入分类器 
    if(!eye_Classifier.load("./haarcascade_eye.xml"))
    { 
        cout<<"Load haarcascade_mcs_eye.xml failed!"<<endl; 
    } 
    vector<string>names;
    vector<Rect> eyeRect;
    Mat image,image_gray;
    Mat storeFrame;
    readFileName(filePath,names);
    for(int i=0;i<names.size();i++)
    {
        eyeRect.clear();
        image = imread(filePath+names[i]); 
        cvtColor(image,image_gray,CV_BGR2GRAY );//转为灰度图 
        equalizeHist(image_gray,image_gray);//直方图均衡化，增加对比度方便处理  
        eye_Classifier.detectMultiScale( image_gray, eyeRect, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );//检测 
        for (size_t eyeIdx = 0;eyeIdx < eyeRect.size();eyeIdx++)
        {
            //画出检测到的位置 
            storeFrame=image(eyeRect[eyeIdx]);
            imwrite("./detect/"+to_string(rand()+344)+".jpg",storeFrame);
            rectangle(image,eyeRect[eyeIdx],Scalar(0,0,255)); 
        }  
        imshow("detect",image); 
        waitKey(999); 
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