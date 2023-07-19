// #define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

template <int BlockDim, int WARP_SZ, int DUP_REM>
__device__ __forceinline__ void BFS_BlockKernelB(int *__restrict__ devNode,
												 int *__restrict__ devEdge,
												 dist_t *__restrict__ devDistance,
												 int *__restrict__ F1,
												 int *__restrict__ F2,
												 int *__restrict__ devF2,
												 int *__restrict__ F2SizePtr,
												 int frontierSize, int level,
												 volatile long long int *HashTable, bool *devAdjMatrix)
{

	int Queue[REG_QUEUE];
	int founds = 0;

	for (int t = Tid >> _Log2<WARP_SZ>::VALUE; t<frontierSize; t += BlockDim>> _Log2<WARP_SZ>::VALUE)
	{
		const int index = F1[t];
		const int start = devNode[index];
		const int end = devNode[index + 1];

		EdgeVisit<BlockDim, WARP_SZ, DUP_REM>(devEdge, devDistance, F2, F2SizePtr, start, end, Queue, founds, level, HashTable, index, devAdjMatrix);
	}

	FrontierWrite::Write<BlockDim, FrontierWrite::SIMPLE>(F2, F2SizePtr, Queue, founds);
}

#define fun(a) BFS_BlockKernelB<BLOCKDIM, (a), DUP_REM>(devNodes, devEdges, devDistance, devF1Pointer, devF2Pointer, NULL, F2SizePtr, frontierSize, level, NULL, devAdjMatrix);

template <int DUP_REM>
__global__ void BFS_BlockKernel(int *__restrict__ devNodes,
								int *__restrict__ devEdges,
								dist_t *__restrict__ devDistance,
								int *__restrict__ devF1,
								int *__restrict__ devF2,
								const int graphVertices,
								const int graphEdges, bool *devAdjMatrix, int run)
{

	//shared variables
	__shared__ int frontierSize;
	__shared__ int visitedNodes;
	__shared__ int level;
	__shared__ int devF2SizeBlock[4];
	// __shared__ int tmpVisited;
	__shared__ int size;
	__shared__ int decDeepness;

	//register variables
	int *devF1Pointer;
	int *devF2Pointer;
	int *F2SizePtr;
	// int size;
	// int decDeepness;

	if (run == 0)
	{
		if (Tid < 4)
			devF2SizeBlock[Tid] = 0;

		if (Tid == BLOCKDIM - 1)
		{
			level = 1;
			visitedNodes = 0;
			frontierSize = 1;
		}

		decDeepness = DEEPNESS;

		devF1Pointer = &devF1[SHAREDMEMORY * blockIdx.x];
		devF2Pointer = &devF2[SHAREDMEMORY * blockIdx.x];
	}
	else
	{
		if (Tid < 4)
			devF2SizeBlock[Tid] = saveF2BlockSize[4 * blockIdx.x + Tid];

		if (Tid == BLOCKDIM - 1)
		{
			visitedNodes = 0;
			frontierSize = saveFrontier[blockIdx.x];
			level = saveLevels[blockIdx.x];
		}

		decDeepness = DEEPNESS;

		devF1Pointer = savePointerF1[blockIdx.x];
		devF2Pointer = savePointerF2[blockIdx.x];
	}

	__syncthreads();

	while (frontierSize && (devGlobalVisited <= (graphEdges / 100) * decDeepness))
	{
		F2SizePtr = &devF2SizeBlock[level & 3];

		if (Tid == 0)
		{
			size = logValueDevice<BLOCKDIM, MIN_VW, MAX_VW>(frontierSize);
			devF2SizeBlock[(level + 1) & 3] = 0;
			visitedNodes += frontierSize;
			atomicAdd(&devGlobalVisited, frontierSize);
		}

		__syncthreads();

		def_SWITCH(size);

		__syncthreads();

		swapDev(devF1Pointer, devF2Pointer);

		if (Tid == 0)
		{
			level++;
			frontierSize = F2SizePtr[0];
		}

		__syncthreads();
	}

	//save F2 status after the run
	if (Tid < 4)
		saveF2BlockSize[4 * blockIdx.x + Tid] = devF2SizeBlock[Tid];

	//save general status after the run
	if (Tid == 5)
	{
		//save frontier
		saveFrontier[blockIdx.x] = frontierSize;
		//save level
		saveLevels[blockIdx.x] = level;

		//save pointer
		savePointerF1[blockIdx.x] = devF1Pointer;
		savePointerF2[blockIdx.x] = devF2Pointer;

		//count number of recursion in while loop per block
		// atomicAdd(&devGlobalLevel, level);

		//count number of active block in execution
		// if (visitedNodes > 0)
		// 	atomicAdd(&devActiveBlocks, 1);
	}
}

#undef fun
