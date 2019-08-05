#ifndef COMMON_INCLUDES
#define COMMON_INCLUDES

#include <assert.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <ctype.h>
#include <array>
#include <vector>
#include <queue>
#include <stack>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <limits>
#include <sstream>
#include <time.h>
#include <string>
#include <fstream>
#include <cstring>
#include <cmath>
#include <time.h>
#include <dirent.h>
#include <cfloat>
#include <cstring>
#include <complex>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>


static void getFlatGridBlock(unsigned long size, dim3 &grid, dim3 &block) {
  if(2147483647 > size){
    grid.x = size;
  }
  else if((unsigned long) 2147483647 * 1024 > size){
    grid.x = 2147483647;
    block.x = 1024;
    while(block.x * grid.x > size){
      block.x--;
    }
    block.x++;
  }
  else{
    grid.x = 65535;
    block.x = 1024;
    grid.y = 1;
    while(grid.x * grid.y * block.x < size){
      grid.y++;
    }
  }
}
static void getGrid(unsigned long size, dim3 &grid) {
  if(2147483647 > size){
    grid.x = size;
  }
  else{
    grid.x = 65535;
    grid.y = 1;
    while(grid.x * grid.y * grid.y < size){
      grid.y++;
    }
  }
}


#endif /* COMMON_INCLUDES */
