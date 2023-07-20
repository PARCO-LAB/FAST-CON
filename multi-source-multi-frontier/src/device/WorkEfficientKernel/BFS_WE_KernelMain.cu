
template<int BlockDim, int WARP_SZ, bool SYNC, bool DUP_REM>
__GLOBAL_DEVICE__ void NAME1 (	 	 int* __restrict__		devNode,
								     int* __restrict__		devEdge,
								  dist_t* __restrict__ 		colors,
								     int* __restrict__ 		devF1,
								     int* __restrict__	 	devF2,
							   const int devF1Size,
							   const int level) {

	int* devF2SizePrt = &devF2Size[level & 3];	// mod 4
	if (blockIdx.x == 0 && Tid == 0)
		devF2Size[(level + 1) & 3] = 0;

	volatile long long int* HashTable;
	if (DUP_REM)
		HashTable = (volatile long long int*) SMem;

	int founds = 0;
	int Queue[REG_QUEUE];

	const int VirtualID = (blockIdx.x * BlockDim + Tid) >> _Log2<WARP_SZ>::VALUE;
	const int Stride = gridDim.x * (BlockDim >> _Log2<WARP_SZ>::VALUE);

	if (!devFoundCommon) {
#if SAFE == 0
		for (int t = VirtualID; t < devF1Size; t += Stride) {
			const int index = devF1[t];
			const int start = devNode[index];
			int end = devNode[index + 1];

			EdgeVisit<BlockDim, WARP_SZ, DUP_REM>(devEdge, colors, devF2, devF2SizePrt, start, end, Queue, founds, level, HashTable, index);
		}
#else
		const int size = ceilf(__fdividef(devF1Size, gridDim.x));
		const int maxLoop = (size + BlockDim / WARP_SZ - 1) >> (_Log2<BlockDim>::VALUE - _Log2<WARP_SZ>::VALUE);

		for (int t = VirtualID, loop = 0; loop < maxLoop; t += Stride, loop++) {
			int index, start, end;
			if (t < devF1Size) {
				index = devF1[t];
				start = devNode[index];
				end = devNode[index + 1];

				DynamicParallelism<BlockDim, WARP_SZ, 0>(devEdge, colors, devF2, start, end, level, index);
			} else
				end = INT_MIN;

			EdgeVisit<BlockDim, WARP_SZ, DUP_REM>(devEdge, colors, devF2, devF2SizePrt, start, end, Queue, founds, level, HashTable, index);
		}
#endif
	}
	if (DUP_REM && (STORE_MODE == FrontierWrite::SHARED_WARP || STORE_MODE == FrontierWrite::SHARED_BLOCK))
		__syncthreads();
	FrontierWrite::Write<BlockDim, FrontierWrite::SIMPLE>(devF2, devF2SizePrt, Queue, founds);
}



template<int BlockDim, int WARP_SZ, bool SYNC, bool DUP_REM>
__GLOBAL_DEVICE__ void NAME1B (	 	 int* __restrict__		devNode,
								     int* __restrict__		devEdge,
								  dist_t* __restrict__ 		colors,
								     int* __restrict__ 		devF1,
								     int* __restrict__	 	devF2,
							   const int devF1Size,
							   const int level) {

	int* devF2SizePrt = &devF2Size[level & 3];	// mod 4
	if (blockIdx.x == 0 && Tid == 0)
		devF2Size[(level + 1) & 3] = 0;

	volatile long long int* HashTable;
	if (DUP_REM)
		HashTable = (volatile long long int*) SMem;

	int* SH_start = (int*) SMem;
	int* SH_end = ((int*) SMem) + BlockDim;
	const int VWarpID = Tid >> _Log2<WARP_SZ>::VALUE;
	SH_start += VWarpID * WARP_SZ;
	SH_end += VWarpID * WARP_SZ;

	int founds = 0;
	int Queue[REG_QUEUE];
	const int VlocalID = Tid & _Mod2<WARP_SZ>::VALUE;
	const unsigned CSIZE = WARP_SZ > 32 ? BlockDim : 32;

    const int size = (devF1Size + CSIZE - 1) & ~(CSIZE - 1);   // ((devF1Size + 32 - 1) / 32) * 32;
    const int Stride = gridDim.x * BlockDim;

    for (int ID = blockIdx.x * BlockDim + Tid; ID < size; ID += Stride) {
		int Tstart, Tend;
		if (ID < devF1Size) {
			int index = devF1[ID];
			Tstart = devNode[index];
			Tend = devNode[index + 1];

			DynamicParallelism<BlockDim, WARP_SZ, 1>(devEdge, colors, devF2, Tstart, Tend, level, index);
		} else
			Tend = INT_MIN;

		if (WARP_SZ > 32) {
			__syncthreads();
			SH_start[VlocalID] = Tstart;
			SH_end[VlocalID] = Tend;
			__syncthreads();
			for (int i = 0; i < WARP_SZ; i++) {
				int start = SH_start[i];
				int end = SH_end[i];
				EdgeVisit<BlockDim, WARP_SZ, DUP_REM>(devEdge, colors, devF2, devF2SizePrt, start, end, Queue, founds, level, HashTable, index);
			}
		}
		else {
			for (int i = 0; i < WARP_SZ; i++) {
				int start = __shfl(Tstart, i, WARP_SZ);
				int end = __shfl(Tend, i, WARP_SZ);
                EdgeVisit<BlockDim, WARP_SZ, DUP_REM>(devEdge, colors, devF2, devF2SizePrt, start, end, Queue, founds, level, HashTable, index);
            }
		}
	}

	if (DUP_REM && (STORE_MODE == FrontierWrite::SHARED_WARP || STORE_MODE == FrontierWrite::SHARED_BLOCK))
		__syncthreads();
	FrontierWrite::Write<BlockDim, FrontierWrite::SIMPLE>(devF2, devF2SizePrt, Queue, founds);
}
