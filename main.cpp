#include"MYSVM.h"
int main()
{
    MYSVM svm;
    svm.svmTrain("./svm_po/","./svm_ne/",80,80);
    svm.svmPredict("./test/");
    //svm.svmPredictPic("./test1/",2);
    //svm.svmPredictVideo("./a.avi",1,1);
    //svm.makeTrainSet("./2002/08/08/big/");
    return 0;
}
