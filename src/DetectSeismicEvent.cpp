#include "common_includes.h"
#include "Unity.cuh"
#include "AnomalyDetector.cuh"
#include "MatlabDataArray.hpp"
#include "MatlabEngine.hpp"


/*
NOTE: CUDA convolution will only be faster than matlab if matrices are large
*/

int main(int argc, char* argv[]){
  time_t start = time_t(nullptr);
  std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr = matlab::engine::startMATLAB();
  matlab::data::ArrayFactory factory;
  std::cout <<"MATLAB engine is running after "<<difftime(time_t(nullptr),start) <<" seconds"<<std::endl;
  start = time_t(nullptr);

  matlabPtr->eval(u"load data/seis_data.mat");
  matlab::data::TypedArray<double> seismicDataMat = matlabPtr->getVariable(u"seis_data");
  int numElements = seismicDataMat.getNumberOfElements();
  double* seismicData_host = new double[numElements]();
  jax::Unity<double>* seismicData = new jax::Unity<double>(seismicData_host, numElements, jax::cpu);
  double max=-DBL_MAX,min=DBL_MAX;

  for(int i = 0; i < numElements; ++i){
    double value = seismicDataMat[i];
    if(value > max) max = value;
    if(value < min) min = value;
  }
  for(int i = 0; i < numElements; ++i){
    seismicDataMat[i] = (seismicDataMat[i] - min)/(max - min);
    seismicData->host[i] = seismicDataMat[i];
  }
  matlabPtr->setVariable(u"seis_data",seismicDataMat);

  jaxdsp::SlidingWindow slidingWindow;
  slidingWindow.setShortTerm(25);
  slidingWindow.setLongTerm(50);
  std::vector<jax::Unity<double2>*> orderedMoments;
  jax::Unity<double>* anomalies;
  jax::Unity<double2>* kurtosis;
  for(int i = 0; i < 5; ++i){
    slidingWindow.setOrder(i+1);
    jax::Unity<double2>* moments;
    slidingWindow.detectAnomaly(seismicData,moments,anomalies,kurtosis);
    orderedMoments.push_back(moments);
  }
  matlab::data::TypedArray<double> momentMat1 = factory.createArray<double>({numElements - slidingWindow.longTermWindow + 1,2});
  matlab::data::TypedArray<double> momentMat2 = factory.createArray<double>({numElements - slidingWindow.longTermWindow + 1,2});
  matlab::data::TypedArray<double> momentMat3 = factory.createArray<double>({numElements - slidingWindow.longTermWindow + 1,2});
  matlab::data::TypedArray<double> momentMat4 = factory.createArray<double>({numElements - slidingWindow.longTermWindow + 1,2});
  matlab::data::TypedArray<double> momentMat5 = factory.createArray<double>({numElements - slidingWindow.longTermWindow + 1,2});
  matlab::data::TypedArray<double> anomaliesMat = factory.createArray<double>({numElements - slidingWindow.longTermWindow + 1,1});
  matlab::data::TypedArray<double> kurtosisMat = factory.createArray<double>({numElements - slidingWindow.longTermWindow + 1,2});
  for(int i = 0; i < numElements - slidingWindow.longTermWindow + 1; ++i){
    momentMat1[i][0] = orderedMoments[0]->host[i].x;
    momentMat1[i][1] = orderedMoments[0]->host[i].y;
    momentMat2[i][0] = orderedMoments[1]->host[i].x;
    momentMat2[i][1] = orderedMoments[1]->host[i].y;
    momentMat3[i][0] = orderedMoments[2]->host[i].x;
    momentMat3[i][1] = orderedMoments[2]->host[i].y;
    momentMat4[i][0] = orderedMoments[3]->host[i].x;
    momentMat4[i][1] = orderedMoments[3]->host[i].y;
    momentMat5[i][0] = orderedMoments[4]->host[i].x;
    momentMat5[i][1] = orderedMoments[4]->host[i].y;
    anomaliesMat[i] = anomalies->host[i];
    kurtosisMat[i][0] = kurtosis->host[i].x;
    kurtosisMat[i][1] = kurtosis->host[i].y;
  }
  matlabPtr->setVariable(u"momentMat1",momentMat1);
  matlabPtr->setVariable(u"momentMat2",momentMat2);
  matlabPtr->setVariable(u"momentMat3",momentMat3);
  matlabPtr->setVariable(u"momentMat4",momentMat4);
  matlabPtr->setVariable(u"momentMat5",momentMat5);
  matlabPtr->setVariable(u"anomalies",anomaliesMat);
  matlabPtr->setVariable(u"kurtosis",kurtosisMat);


  matlabPtr->eval(u"figure;subplot(5,1,1);plot(momentMat1);title('First Order Statistical Moment');\
  subplot(5,1,2);plot(momentMat2);title('Second Order Statistical Moment');\
  subplot(5,1,3);plot(momentMat3);title('Third Order Statistical Moment');\
  subplot(5,1,4);plot(momentMat4);title('Fourth Order Statistical Moment');\
  subplot(5,1,5);plot(momentMat5);title('Fifth Order Statistical Moment');");

  matlabPtr->eval(u"figure;subplot(3,1,1);plot(seis_data);title('Seismic Data');\
  subplot(3,1,2);plot(anomalies);title('S/L - Kurt ratio');\
  subplot(3,1,3);plot(kurtosis);title('STK & LTK');");

  matlabPtr->feval<void>(u"pause",1000);
  return 0;
}
