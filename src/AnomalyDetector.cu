#include "AnomalyDetector.cuh"

template<typename T>
__global__ void slideWindowDetector(unsigned int shortTermWindow, unsigned int longTermWindow, unsigned int numElements, T* data, double* anomalies, double2* moments, double2* flkclk, unsigned int order){
  int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int globalId = blockId * (blockDim.x * blockDim.y * blockDim.z) +
  (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
  if(globalId < numElements){
    unsigned int ls = shortTermWindow;
    unsigned int ll = longTermWindow;

    double2 mean={0.0,0.0};
    T* reg_data = new T[ll];
    for(int i = globalId; i < ll + globalId; ++i){
      reg_data[i - globalId] = data[i];
      if(ls + globalId > i){
        mean.x += reg_data[i - globalId];
      }
      mean.y += reg_data[i - globalId];
    }
    mean.x /= (double)ls;
    mean.y /= (double)ll;

    double stk = 0.0;
    double ltk = 0.0;
    double2 dev= {0.0,0.0};
    double temp = 0.0;
    double currentValue = 0.0;
    for(int i = 0; i < ll; ++i){
      currentValue = reg_data[i];
      if(ls > i){
        temp = (currentValue - mean.x);
        dev.x += temp*temp;
        stk += temp*temp*temp*temp;
      }
      temp = (currentValue - mean.y);
      dev.y += temp*temp;
      ltk += temp*temp*temp*temp;
    }
    dev.x /= ((double)ls-1);
    dev.y /= ((double)ll-1);
    stk /= (((double)ls - 1)*dev.x*dev.x);
    ltk /= (((double)ll - 1)*dev.y*dev.y);

    double ratio = stk/(ltk + 0.0000000000001);

    dev.x = sqrt(dev.x);
    dev.y = sqrt(dev.y);

    unsigned int reg_order = order;
    double2 rawMoments[5] = {{0.0,0.0},{0.0,0.0},{0.0,0.0},{0.0,0.0},{0.0,0.0}};
    for(int o = 0; o < reg_order; ++o){
      for(int i = 0; i < ll; ++i){
        if(ls > i){
          rawMoments[o].x += pow(reg_data[i],o+1)*(1.0/dev.x*sqrt(2*M_PI))*exp(-1*(reg_data[i] - mean.x)*(reg_data[i] - mean.x)/(2*dev.x*dev.x));
        }
        rawMoments[o].y += pow(reg_data[i],o+1)*(1.0/dev.y*sqrt(2*M_PI))*exp(-1*(reg_data[i] - mean.y)*(reg_data[i] - mean.y)/(2*dev.y*dev.y));
      }
    }

    double2 moment = {0.0,0.0};
    if(reg_order == 1) moment = rawMoments[0];
    else if(reg_order == 2){
      moment = {rawMoments[1].x - (rawMoments[0].x*rawMoments[0].x),rawMoments[1].y - (rawMoments[0].y*rawMoments[0].y)};
    }
    else if(reg_order == 3){
      moment = {rawMoments[2].x - (3*rawMoments[0].x*rawMoments[1].x) +
      (2*rawMoments[0].x*rawMoments[0].x*rawMoments[0].x),
      rawMoments[2].y - (3*rawMoments[0].y*rawMoments[1].y) +
      (2*rawMoments[0].y*rawMoments[0].y*rawMoments[0].y)};
    }
    else if(reg_order == 4){
      moment = {rawMoments[3].x - (4*rawMoments[0].x*rawMoments[2].x) +
      (6*rawMoments[0].x*rawMoments[0].x*rawMoments[1].x) -
      (3*rawMoments[0].x*rawMoments[0].x*rawMoments[0].x*rawMoments[0].x),
      rawMoments[3].y - (4*rawMoments[0].y*rawMoments[2].y) +
      (6*rawMoments[0].y*rawMoments[0].y*rawMoments[1].y) -
      (3*rawMoments[0].y*rawMoments[0].y*rawMoments[0].y*rawMoments[0].y)};
    }
    else if(reg_order == 5){
      moment = {rawMoments[4].x - (5*rawMoments[0].x*rawMoments[3].x) +
      (10*rawMoments[0].x*rawMoments[0].x*rawMoments[2].x) -
      (10*rawMoments[0].x*rawMoments[0].x*rawMoments[0].x*rawMoments[1].x),
      rawMoments[4].y - (5*rawMoments[0].y*rawMoments[3].y) +
      (10*rawMoments[0].y*rawMoments[0].y*rawMoments[2].y) -
      (10*rawMoments[0].y*rawMoments[0].y*rawMoments[0].y*rawMoments[1].y)};
    }
    else{
      delete[] reg_data;
      printf("bad order\n");
      asm("trap;");
    }
    flkclk[globalId] = {stk,ltk};
    anomalies[globalId] = ratio;
    moments[globalId] = moment;
    delete[] reg_data;
  }
}


jaxdsp::SlidingWindow::SlidingWindow(){
  this->shortTermWindow = 1;
  this->longTermWindow = 2;
  this->order = 1;
}

jaxdsp::SlidingWindow::~SlidingWindow(){

}

void jaxdsp::SlidingWindow::setShortTerm(const unsigned int &shortTermWindow){
  this->shortTermWindow = shortTermWindow;
}
void jaxdsp::SlidingWindow::setLongTerm(const unsigned int &longTermWindow){
  this->longTermWindow = longTermWindow;
}
void jaxdsp::SlidingWindow::setOrder(const unsigned int &order){
  this->order = order;
}

void jaxdsp::SlidingWindow::detectAnomaly(jax::Unity<double>* data, jax::Unity<double2>* &moments, jax::Unity<double>* &anomalies, jax::Unity<double2>* &kurtValues){
  assert(data != NULL);
  data->transferMemoryTo(jax::gpu);

  double* anomalies_host = new double[data->numElements - this->longTermWindow + 1]();
  anomalies = new jax::Unity<double>(anomalies_host,data->numElements - this->longTermWindow + 1,jax::cpu);
  double2* moments_host = new double2[data->numElements - this->longTermWindow + 1]();
  moments = new jax::Unity<double2>(moments_host,data->numElements - this->longTermWindow + 1,jax::cpu);
  double2* kurts_host = new double2[data->numElements - this->longTermWindow + 1]();
  kurtValues = new jax::Unity<double2>(kurts_host, data->numElements - this->longTermWindow + 1, jax::cpu);

  kurtValues->transferMemoryTo(jax::gpu);
  anomalies->transferMemoryTo(jax::gpu);
  moments->transferMemoryTo(jax::gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(data->numElements - this->longTermWindow + 1, grid, block);

  slideWindowDetector<<<grid, block>>>(this->shortTermWindow, this->longTermWindow, data->numElements - this->longTermWindow + 1, data->device, anomalies->device, moments->device, kurtValues->device, this->order);
  kurtValues->transferMemoryTo(jax::cpu);
  anomalies->transferMemoryTo(jax::cpu);
  moments->transferMemoryTo(jax::cpu);
}
