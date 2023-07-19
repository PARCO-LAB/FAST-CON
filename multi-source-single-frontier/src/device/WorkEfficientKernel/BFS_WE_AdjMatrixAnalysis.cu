__device__ bool checkedSources[NUM_SOURCES];
__device__ int countTempReachByS;

__global__ void AdjAnalysis(bool *devAdjMatrix)
{
	if (devAdjMatrix[1])
	{
		found_common = 1;
		return;
	}

	if(blockIdx.x > 1)
	{
		if (devAdjMatrix[blockIdx.x*NUM_SOURCES] && !checkedSources[blockIdx.x])
		{
			for (int thread = threadIdx.x; thread < NUM_SOURCES; thread += blockDim.x)
			{
				if (devAdjMatrix[(blockIdx.x * NUM_SOURCES) + thread])
				{
					devAdjMatrix[thread] = 1;
					devAdjMatrix[thread*NUM_SOURCES] = 1;
					countTempReachByS = 1;
				}
			}
			checkedSources[blockIdx.x] = 1;
		}
	}

	if (devAdjMatrix[1])
	{
		found_common = 1;
		countTempReachByS = 0;
		return;
	}
}
