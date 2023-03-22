/**
 * 利用opencv来加载训练的模型
 * date:2023.3.21
 * **/
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <boost/foreach.hpp>
using namespace std;
using namespace cv;
using namespace cv::dnn;

template <typename _tp>
vector<_tp> convertMat2Vector(const Mat &mat)
{
    return (vector<_tp>) (mat.reshape(1,1));
}
int MaxVector(vector<float>& v)
{
    int flag =0;
    float max(0.0);
    for (size_t i = 0; i < v.size(); i++)
    {
        if(v[i]>max)
        {
            max=v[i];
            flag = i;
        }
    }
    return flag;
}
int main(int argc ,char** argv)
{
    string label[] = {"plane","car","bird","cat","deer","dog","frog","horse","ship","truck"};
    
    vector<string> labels(label,label+10);
    BOOST_FOREACH(auto & i,labels)
    {
        cout<< i <<endl;
    }
    string modelFile = "/home/wf/c++_pytorch/torch.onnx";
    string imageFile = "/home/wf/c++_pytorch/data/ship1.jpg";
    dnn::Net net = cv::dnn::readNetFromONNX(modelFile);
    Mat image = imread(imageFile);
    cv::imshow("显示图片",image);
    cv::waitKey(0);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    //用来对图片进行预处理
    //整体像素值减去平均值
    //通过缩放系数对图片像素值进行缩放
    //第一个参数，InputArray image，表示输入的图像，可以是opencv的mat数据类型。
	//第二个参数，scalefactor，这个参数很重要的，如果训练时，是归一化到0-1之间，那么这个参数就应该为0.00390625f （1/256），否则为1.0
	//第三个参数，size，应该与训练时的输入图像尺寸保持一致。
	//第四个参数，mean，这个主要在caffe中用到，caffe中经常会用到训练数据的均值。tf中貌似没有用到均值文件。
	//第五个参数，swapRB，是否交换图像第1个通道和最后一个通道的顺序。
	//第六个参数，crop，如果为true，就是裁剪图像，如果为false，就是等比例放缩图像。
    Mat inputBolb = blobFromImage(image, 0.003906256f, Size(32,32), Scalar(),false,false);
    
    net.setInput(inputBolb); //输入图像

    Mat result = net.forward();

    cout <<result <<endl;
    vector<float> v = convertMat2Vector<float>(result);

    int count = MaxVector(v);

    cout <<  labels[count] << endl;

}