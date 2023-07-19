#include "cudaGraph.cuh"

__device__ int duplicateCounter;
__device__ int devF2Size[4];
__device__ int devGlobalLevel;
__device__ int devGlobalVisited;
__device__ int devActiveBlocks;
__device__ int countFail;
__device__ bool found_common;

//save
__device__ int *savePointerF1[NUM_BLOCKS];
__device__ int *savePointerF2[NUM_BLOCKS];
__device__ int saveFrontier[NUM_BLOCKS];
__device__ int saveLevels[NUM_BLOCKS];
__device__ int saveF2BlockSize[4 * NUM_BLOCKS];

cudaGraph::cudaGraph(Graph &_graph) : graph(_graph)
{
	V = graph.V;
	E = graph.E;

	cudaMalloc(&devOutNodes, (V + 1) * sizeof(int));
	cudaMalloc(&devOutEdges, E * sizeof(int));
	cudaMalloc(&devDistance, V * sizeof(dist_t));
	cudaMalloc(&devAdjMatrix, NUM_SOURCES * NUM_SOURCES * sizeof(bool));

	cudaMemcpy((void **)devOutNodes, graph.OutNodes, (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void **)devOutEdges, graph.OutEdges, E * sizeof(int), cudaMemcpyHostToDevice);

	const int ZERO = 0;
	cudaMemcpyToSymbol(duplicateCounter, &ZERO, sizeof(int));

	cudaError("Graph Allocation");

	// --------------- Frontier Allocation -------------------

	size_t free, total;
	cudaMemGetInfo(&free, &total);
	size_t frontierSize = (free / 2) - 500 * 8192;

	cudaMalloc(&devF1, frontierSize);
	cudaMalloc(&devF2, frontierSize);
	allocFrontierSize = frontierSize / sizeof(int);

	cudaError("Graph Frontier Allocation");
}

#include "WorkEfficientKernel/BFS_WE_Kernels1.cu"
#include "WorkEfficientKernel/BFS_WE_Dynamic.cu"

// ----------------------- GLOBAL SYNCHRONIZATION --------------------------------

#define __GLOBAL_DEVICE__ __global__
#define NAME1 BFS_KernelMainGLOB
#define NAME1B BFS_KernelMainGLOBB

#include "WorkEfficientKernel/BFS_WE_KernelMain.cu"

#undef __GLOBAL_DEVICE__
#undef NAME1
#undef NAME1B

#define __GLOBAL_DEVICE__ __device__
#define NAME1 BFS_KernelMainDEV
#define NAME1B BFS_KernelMainDEVB

#include "WorkEfficientKernel/BFS_WE_KernelMain.cu"

#undef __GLOBAL_DEVICE__
#undef NAME1
#undef NAME1B

// ----------------------------------------------------------------------------------

#include "WorkEfficientKernel/BFS_WE_KernelDispatch.cu"
#include "WorkEfficientKernel/BFS_WE_Block.cu"
#include "WorkEfficientKernel/BFS_WE_AdjMatrixAnalysis.cu"

#include "BFS_WorkEfficient.cu"
