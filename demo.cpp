#include <iostream>
#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api.h"
#include "onnxmodel.h"
using namespace cv;

class HumanDetect:public OnnxModel{
public:
    HumanDetect(std::wstring Path):OnnxModel(Path){};

private:

    virtual std::vector<vector<float> > PreProcess(Mat& Src){
        vector<vector<float> > Result;

        Mat Dst = dnn::blobFromImage(Src, 1.0, Size(640, 640), Scalar());
        vector<float> TempFloat;
        for(int i = 0;i < Dst.total();i++){
            TempFloat.push_back(*(Dst.ptr<float>() + i));
        }
        Result.push_back(TempFloat);

        vector<float> TempScalar = { 640.0f / Src.rows,640.0f / Src.cols };
        Result.push_back(TempScalar);
        return Result;
    }
};
int main(){
    std::wstring Path = L"./model/humandetect.onnx";
    HumanDetect HD(Path);
    cv::VideoCapture cap(0);

    while(cap.isOpened()) {
        cv::Mat temp;
        cap >> temp;
        vector<vector<float> > Result = HD.Predict(temp);
        vector<Rect> ProRes;
        for(int i = 0; i < Result[0].size(); i += 6) {
            //int id = Result[0][i] + 0.5;
            float score = Result[0][i + 1];
            Rect box(Result[0][i + 2], Result[0][i + 3],
                Result[0][i + 4], Result[0][i + 5]);
            if(score > 0.5)rectangle(temp, box, Scalar(255, 0, 0), 1);
        }
        imshow("test", temp);
        waitKey(1);
    }
}