#pragma once

extern __shared__ unsigned char SMem[];

#include "../Util/ptx.cu"
#include "../Util/GlobalWrite.cu"

#define PRIMEQ	2654435769u				// ((sqrt(5)-1)/2) << 32

template<int BlockDim>
__device__ __forceinline__ void DuplicateRemove(const int dest, volatile long long int* HashTable, int* Queue, int &founds) {
	unsigned hash = dest * PRIMEQ;
	hash = (hash & (unsigned)(SMem_Per_Block(BlockDim)/12 - 1)) + (hash & (unsigned)(SMem_Per_Block(BlockDim)/24 - 1));

	int2 toWrite = make_int2(Tid, dest);
	HashTable[hash] = reinterpret_cast<volatile long long int&>(toWrite);
	int2 recover = reinterpret_cast<int2*>( const_cast<long long int*>(HashTable) )[hash];
	if (recover.x == Tid || recover.y != dest)
		Queue[founds++] = dest;
	else if (COUNT_DUP && recover.x != Tid && recover.y == dest)
		atomicAdd(&duplicateCounter, 1);
}


template<int BlockDim, bool DUP_REM>
__device__ __forceinline__ void KVisit(	const int dest, dist_t* colors,
										int* Queue, int& founds, const int level, volatile long long int* HashTable, const int src, bool* devAdjMatrix) {
	#if ATOMICCAS
		if (atomicCAS(&colors[ dest ], (dist_t) -1, level) == (dist_t) -1)
			Queue[founds++] = dest;
		else if (colors[dest] != colors[index])
			devFoundCommon = 1;
	#else

	// Color the node
	if (colors[dest] == (dist_t) -1) {
		colors[dest] = colors[src];
		if (DUP_REM)
			DuplicateRemove<BlockDim>(dest, HashTable, Queue, founds);
		else
			Queue[founds++] = dest;
	}
	// Found a node that already has a color -> Connection found between two sources!
	// Update the adjacency matrix
	else if (colors[dest] != colors[src]) {
		devAdjMatrix[(colors[dest] * NUM_SOURCES) + colors[src]] = 1;
		devAdjMatrix[(colors[src] * NUM_SOURCES) + colors[dest]] = 1;
	}
	#endif
}


template<int BlockDim, int WARP_SZ, bool DUP_REM>
__device__ __forceinline__ void EdgeVisit(	 	   int* __restrict__	devEdge,
												dist_t* __restrict__	colors,
												   int* __restrict__ 	devF2,
												   int* __restrict__	devF2SizePrt,
													int start, int end,
													int* Queue, int& founds, const int level, volatile long long int* HashTable, const int index, bool* devAdjMatrix) {
#if SAFE == 0
	for (int k = start + (Tid & _Mod2<WARP_SZ>::VALUE); k < end; k += WARP_SZ) {
		const int dest = devEdge[k];

		KVisit<BlockDim, DUP_REM>(dest, colors, Queue, founds, level, HashTable, index, devAdjMatrix);
	}
#elif SAFE == 1
	bool flag = true;
	int k = start + (Tid & _Mod2<WARP_SZ>::VALUE);
	while (flag && !devFoundCommon) {
		while (k < end && founds < REG_QUEUE) {
			const int dest = devEdge[k];

			KVisit<BlockDim, DUP_REM>(dest, colors, Queue, founds, level, HashTable, index, devAdjMatrix);
			k += WARP_SZ;
		}
		if (__any(founds >= REG_QUEUE)) {
			FrontierWrite::Write<BlockDim, FrontierWrite::SIMPLE>(devF2, devF2SizePrt, Queue, founds);
			founds = 0;
		} else
			flag = false;
	}
#endif
}
