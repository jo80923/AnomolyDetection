CUDA_INSTALL_PATH := /usr/local/cuda
MATLAB_INSTALL_PATH := /usr/local/MATLAB/R2019a

# CUDA stuff
CXX := g++-5
LINK := nvcc
NVCC  := nvcc

# Includes
INCLUDES = -I. -I./include -I/usr/local/cuda/include -I/usr/local/MATLAB/R2019a/extern/include

# Common flags
COMMONFLAGS += ${INCLUDES}
CXXFLAGS += ${COMMONFLAGS}
CXXFLAGS += -Wall -std=c++11
# compute_<#> and sm_<#> will need to change depending on the device
# if this is not done you will receive a no kernel image is availabe error
NVCCFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -std=c++11 -gencode=arch=compute_61,code=sm_61

LIB :=  -L/usr/local/cuda/lib64 -lcublas -lcuda -lcudart -L/usr/local/MATLAB/R2019a/extern/bin/glnxa64/ -lMatlabDataArray -lMatlabEngine


SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin

_OBJS = AnomalyDetector.cu.o
_OBJS += DetectSeismicEvent.cpp.o

OBJS = ${patsubst %, ${OBJDIR}/%, ${_OBJS}}

TARGET = DetectSeismicEvent

LINKLINE = ${LINK} -gencode=arch=compute_61,code=sm_61 ${OBJS} ${LIB} -o ${BINDIR}/${TARGET}

.SUFFIXES: .cpp .cu .o

all: ${BINDIR}/${TARGET}

$(OBJDIR):
	    -mkdir -p $(OBJDIR)
$(BINDIR):
	    -mkdir -p $(BINDIR)

${OBJDIR}/%.cu.o: ${SRCDIR}/%.cu
	${NVCC} ${INCLUDES} ${NVCCFLAGS} -dc $< -o $@

${OBJDIR}/%.cpp.o: ${SRCDIR}/%.cpp
	${CXX} ${INCLUDES} ${CXXFLAGS} -c $< -o $@

${BINDIR}/%: ${OBJS} Makefile
	${LINKLINE}

clean:
	rm -f bin/*
	rm -f obj/*
