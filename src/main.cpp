#include"MYSVM.h"
#include"CutPic.h"
#include<opencv2/opencv.hpp>
#include<string>
using namespace cv;
using namespace std;
int main()
{
    CutPic p;
    //p.cutPic(src,"./test");
    //p.createPicFile("../qrpic/","../cutpic/");
    //p.cutVideo("../video/src.mp4",150,150,1);
    MYSVM svm(160,160);
    //svm.svmTrain("../positive/","../negetive/",160,160);
    //svm.saveSvmXml("../model/qrcode_real.xml");
    svm.loadModel("../model/qrcode_real.xml");
    
    svm.svmPredict("../test/");
    svm.svmPredictVideo("../video/src.mp4",4);
    //svm.svmPredictPic("../testpic/",1);
    return 0;
}
