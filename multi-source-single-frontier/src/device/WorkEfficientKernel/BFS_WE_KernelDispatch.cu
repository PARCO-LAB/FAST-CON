#pragma once

#include "../Util/GlobalSync.cu"

template <int AVAILABLE_THREAD, int MIN_VALUE, int MAX_VALUE>
__device__ __forceinline__ int logValueDevice(int Value);
__device__ __forceinline__ void swapDev(int *&A, int *&B);

#define fun(a) BFS_KernelMainDEV<BlockDim, (a), true, DUP_REM>(devNodes, devEdges, devDistance, devF1, devF2, FrontierSize, level);

#define funB(a) BFS_KernelMainDEVB<BLOCKDIM, (a), true, DUP_REM>(devNodes, devEdges, devDistance, devF1, devF2, FrontierSize, level);

template <int BlockDim, bool DUP_REM>
__global__ void BFSDispath(int *__restrict__ devNodes,
													 int *__restrict__ devEdges,
													 dist_t *__restrict__ devDistance,
													 int *devF1,
													 int *devF2)
{
	int FrontierSize = 1, level = 1;
	do
	{
		int size = logValueDevice<MAX_CONCURR_TH, MIN_VW, MAX_VW>(FrontierSize);

		if (size >= 8 && FrontierSize > MAX_CONCURR_TH)
		{
			def_SWITCHB(size);
		}
		else
		{
			def_SWITCH(size);
		}

		GlobalSync();
		FrontierSize = devF2Size[level & 3];
		swapDev(devF1, devF2);
		level++;
	} while (FrontierSize);
}

#undef fun
#undef funB

__device__ __forceinline__ void swapDev(int *&A, int *&B)
{
	int *temp = A; // frontiers swap
	A = B;
	B = temp;
}

template <int AVAILABLE_THREAD, int MIN_VALUE, int MAX_VALUE>
__device__ __forceinline__ int logValueDevice(int Value)
{
	int logSize = 31 - __clz(AVAILABLE_THREAD / Value);
	if (logSize < _Log2<MIN_VALUE>::VALUE)
		logSize = _Log2<MIN_VALUE>::VALUE;
	if (logSize > _Log2<MAX_VALUE>::VALUE)
		logSize = _Log2<MAX_VALUE>::VALUE;
	return logSize;
}
