#ifndef ANOMALYDETECTOR_CUH
#define ANOMALYDETECTOR_CUH

#include "common_includes.h"
#include "Unity.cuh"




template<typename T>
__global__ void slideWindowDetector(unsigned int shortTermWindow, unsigned int longTermWindow, unsigned int numElements, T* data, double* anomalies, unsigned int order);

namespace jaxdsp{

  class SlidingWindow{

  public:

    unsigned int shortTermWindow;
    unsigned int longTermWindow;
    unsigned int order;

    SlidingWindow();
    ~SlidingWindow();
    void setLongTerm(const unsigned int &longTermWindow);
    void setShortTerm(const unsigned int &shortTermWindow);
    void setOrder(const unsigned int &order);

    void detectAnomaly(jax::Unity<double>* data, jax::Unity<double2>* &moments, jax::Unity<double>* &anomalies, jax::Unity<double2>* &kurtValues);
  };
}




#endif /* ANOMALYDETECTOR_CUH */
