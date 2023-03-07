#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api.h"
#include "opencv2/opencv.hpp"

#ifdef _WIN32
typedef std::wstring ortstring;
#else
typedef std::string ortstring;
#endif

#define NUM_THREADS 1

using namespace std;

class OnnxModel{
public:
    OnnxModel(ortstring Path){
        this->Enviro = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Defalut");
        this->SessOpts = Ort::SessionOptions();
        this->SessOpts.SetIntraOpNumThreads(NUM_THREADS);

        Ort::AllocatorWithDefaultOptions Allocator;
        this->Sess = new Ort::Session(Enviro, Path.c_str(), SessOpts);

        //获取输入、输出的数量
        this->InputCount = this->Sess->GetInputCount();
        this->OutputCount = this->Sess->GetOutputCount();

        //获取onnx输入名
        for(int i = 0;i < InputCount;i++){
            InputName.push_back(this->Sess->GetInputName(i, Allocator));
        }

        //获取onnx输出名
        for(int i = 0;i < OutputCount;i++){
            OutputName.push_back(this->Sess->GetOutputName(i, Allocator));
        }

        //获取onnx输入格式
        std::vector<std::vector<int64_t> > InputDims;
        for(int i = 0;i < this->InputCount;i++){
            std::vector<int64_t> Shape = this->Sess->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

            //不定长batchsize改成定长且为1
            Shape[0] = 1;
            this->InputDims.push_back(Shape);
        }
    }


    //对预处理完的图像进行推理
    std::vector<vector<float> > Predict(cv::Mat Src){
        std::vector<vector<float> >InputData = this->PreProcess(Src);
        Ort::MemoryInfo MemInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        //创建输入tensor
        std::vector<Ort::Value> InputTensor;
        for(int i = 0;i < this->InputCount;i++){
            std::vector<int64_t>& InputDim = this->InputDims[i];
            InputTensor.push_back(Ort::Value::CreateTensor<float>(MemInfo, InputData[i].data(),
                InputData[i].size(), InputDim.data(), InputDim.size()));
        }

        //进行推理
        auto OutputTensor = this->Sess->Run(Ort::RunOptions{nullptr}, this->InputName.data(),
            InputTensor.data(), this->InputCount, this->OutputName.data(), this->OutputCount);

        //获取推理结果
        std::vector<vector<float> > OutputDatas;
        for(int i = 0;i < this->OutputCount;i++){
            float* TempFloat = OutputTensor[i].GetTensorMutableData<float>();
            size_t TempLen = OutputTensor[i].GetTensorTypeAndShapeInfo().GetElementCount();
            std::vector<float> OutputData;
            for(int j = 0;j < TempLen;j++){
                OutputData.push_back(TempFloat[j]);
            }
            OutputDatas.push_back(OutputData);
        }
        return OutputDatas;
    }

protected:
    //对图像预处理
    virtual std::vector<vector<float> > PreProcess(cv::Mat& Src) = 0;

private:
    Ort::Env Enviro;
    Ort::SessionOptions SessOpts;
    Ort::Session* Sess{nullptr};
    size_t InputCount;
    size_t OutputCount;
    std::vector<std::vector<int64_t> > InputDims;
    std::vector<const char*> InputName;
    std::vector<const char*> OutputName;
    OnnxModel() = delete;
};