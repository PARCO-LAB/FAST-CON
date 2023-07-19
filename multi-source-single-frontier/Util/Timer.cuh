#pragma once

#include <sys/time.h>
#include <iostream>
#include <string>
#include <iomanip>		// set precision cout
#include "cudaUtil.cuh"	// getLastError, COLOR

/*#if __cplusplus > 199711L || __GXX_EXPERIMENTAL_CXX0X__
	#include <chrono>
#endif*/
#if defined(__NVCC__)
	#include <cuda_runtime.h>
#endif

#define getTimeE(msg)		getTime((msg), __FILE__, __LINE__)



enum timerType { HOST, DEVICE, DEVICE_H };

template<timerType type>
class Timer {
	private:
	timeval startTime, endTime;
	cudaEvent_t startTimeCuda, stopTimeCuda;
	
	int P, W;
		
	public:
		Timer(int _P = 1, int _W = 1);
		void start();
		void stop();
		float duration();
		void getTime(std::string str);
		void getTime(std::string str, const char *file, const int line);
};

#include "Timer.cu"
