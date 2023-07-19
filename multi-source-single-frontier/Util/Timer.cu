//#include "Timer.cuh"

/*#if __cplusplus > 199711L || __GXX_EXPERIMENTAL_CXX0X__
	template<>
	void Timer<HOST>::start() {
		startTime = std::chrono::system_clock::now();
	}

	template<>
	void Timer<HOST>::stop() {
		endTime = std::chrono::system_clock::now();
	}

	template<>
	float Timer<HOST>::duration() {
		return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() * 1000;
	}
#else*/


	template<>
	inline void Timer<HOST>::start() {
		gettimeofday(&startTime, NULL);
	}

	template<>
	inline void Timer<HOST>::stop() {
		gettimeofday(&endTime, NULL);
	}

	template<>
	inline float Timer<HOST>::duration() {
		int secs(endTime.tv_sec - startTime.tv_sec);
		int usecs(endTime.tv_usec - startTime.tv_usec);

		if(usecs < 0) {
			--secs;
			usecs += 1000000;
		}

		return (secs * 1000 + usecs / 1000.0);
		//return static_cast<int>(secs * 1000 + usecs / 1000.0 + 0.5);
	}
//#endif

	template<>
	inline void Timer<HOST>::getTime(std::string str) {
		Timer<HOST>::stop();
		std::cout << Color::FG_L_BLUE << std::setw(W) << str
				<< '\t' << std::fixed << std::setprecision(P) << duration() << " ms" << Color::FG_DEFAULT << std::endl;
	}

	template<>
	inline Timer<HOST>::Timer(int _P, int _W) : P(_P), W(_W) {
		//Timer<HOST>::start();
	}

#if defined(__NVCC__)

template<>
void Timer<DEVICE>::start() {
	cudaEventRecord(startTimeCuda, 0);
}

template<>
void Timer<DEVICE>::stop() {
	cudaEventRecord(stopTimeCuda, 0);
	cudaEventSynchronize(stopTimeCuda);
}

template<>
float Timer<DEVICE>::duration() {
	float time;
	cudaEventElapsedTime(&time, startTimeCuda, stopTimeCuda);
	return time;
}

template<>
void Timer<DEVICE>::getTime(std::string str) {
	Timer<DEVICE>::stop();
	std::cout << Color::FG_L_RED << std::setw(W) << str
			  << '\t' << std::fixed << std::setprecision(P) << duration() << " ms" << Color::FG_DEFAULT << std::endl;
}

template<>
void Timer<DEVICE>::getTime(std::string str, const char *file, const int line) {
	Timer<DEVICE>::getTime(str);
	cudaDeviceSynchronize();
	cudaUtil::__getLastCudaError(str.c_str(), file, line);
}

template<>
Timer<DEVICE>::Timer(int _P, int _W) : P(_P), W(_W) {
	cudaEventCreate(&startTimeCuda);
	cudaEventCreate(&stopTimeCuda);
	//Timer<DEVICE>::start();
}


//---------------------------------------------------------------------------------
/*
template<>
void Timer<DEVICE_H>::start() {
	gettimeofday(&startTime, NULL);
}

template<>
void Timer<DEVICE_H>::stop() {
	cudaDeviceSynchronize();
	gettimeofday(&endTime, NULL);
}

template<>
float Timer<DEVICE_H>::duration() {
	int secs(endTime.tv_sec - startTime.tv_sec);
	int usecs(endTime.tv_usec - startTime.tv_usec);

	if(usecs < 0) {
		--secs;
		usecs += 1000000;
	}

	return (secs * 1000 + usecs / 1000.0);
	//return static_cast<int>(secs * 1000 + usecs / 1000.0 + 0.5);
}

template<>
void Timer<DEVICE_H>::getTime(std::string str) {
	Timer<DEVICE_H>::stop();
	std::cout << std::setw(W) << str
		 << '\t' << std::fixed << std::setprecision(P) << duration() << " ms\n";
}

template<>
Timer<DEVICE_H>::Timer(int _P, int _W) : P(_P), W(_W) {
	Timer<DEVICE_H>::start();
}

template<>
void Timer<DEVICE_H>::getTime(std::string str, const char *file, const int line) {
	Timer<DEVICE_H>::getTime(str);
	cudaDeviceSynchronize();
	fUtil::__getLastCudaError(str.c_str(), file, line);
}*/

#endif
