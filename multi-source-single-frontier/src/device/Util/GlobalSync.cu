#pragma once

#include <cub/cub.cuh>

__device__ unsigned int GSync[1024];

__global__ void GReset() {
	if (Tid < 1024)
		GSync[Tid] = 0;
}

__device__  __forceinline__ void GlobalSync() {
	volatile unsigned *VolatilePtr = GSync;
	__syncthreads();

	if (blockIdx.x == 0) {
		if (Tid == 0)
			VolatilePtr[blockIdx.x] = 1;
		//__syncthreads();

		if (Tid < gridDim.x)
			while ( cub::ThreadLoad<cub::LOAD_CG>(GSync + Tid) == 0 );

		__syncthreads();

		if (Tid < gridDim.x)
			VolatilePtr[Tid] = 0;
	}
	else {
		if (Tid == 0) {
			VolatilePtr[blockIdx.x] = 1;
			while (cub::ThreadLoad<cub::LOAD_CG>(GSync + blockIdx.x) == 1);
		}
		__syncthreads();
	}
}
