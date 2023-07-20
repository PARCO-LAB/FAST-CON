#pragma once

long long int totalFrontierNodes;
long long int totalFrontierEdges;

void cudaGraph::cudaFASTCON_N(int nof_tests) {
	srand(time(NULL));
	if (DUPLICATE_REMOVE)
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	long long int totalVisitedEdges = 0, totalVisitedNodes = 0, diameter = 0;
	totalFrontierNodes = 0, totalFrontierEdges = 0;

	int* hostDistance = new int[V];
	Timer<DEVICE> TM;
	double totalTime = 0;

	int stConnectionsFound = 0;
	for (int i = 0; i < N_OF_TESTS; i++) {

		// Select random sources
		int *sources = new int[NUM_SOURCES];
		for (int j = 0; j < NUM_SOURCES; j++)
			sources[j] = N_OF_TESTS == 1 ? 0 : mt::randf_co() * V;

		totalVisitedNodes += graph.visitedNodes();
		totalVisitedEdges += graph.visitedEdges();
		diameter += graph.getMaxDistance();

		this->Reset(sources, NUM_SOURCES);

		TM.start();
		this->cudaFASTCON();
		TM.stop();

		cudaError("BFS Kernel N");

		float time = TM.duration();
		totalTime += time;

		int foundCommon = 0;
		cudaMemcpyFromSymbol(&foundCommon, devFoundCommon, sizeof(int));
		if (foundCommon) {
			std::cout << "Found ST-Connection\n";
			stConnectionsFound++;
		}

		if (nof_tests > 1)
			std::cout 	<< "iter: " << std::left << std::setw(10) << i << "time: " << std::setw(10) << time << "source: " << sources[0] << " target: " << sources[1] << std::endl;
		else
			cudaUtil::Compare(hostDistance, colors, V, "Distance Check", 1);
	}

	std::cout	<< std::setprecision(2) << std::fixed << std::endl
				<< "\t      Number of TESTS: " << nof_tests << std::endl
				<< "\t    Number of SOURCES: " << NUM_SOURCES << std::endl
				<< "\t ST-Connections found: " << stConnectionsFound << std::endl
				<< "\t            Avg. Time: " << totalTime / nof_tests << " ms" << std::endl;

	if (COUNT_DUP && nof_tests == 1) {
		int duplicates;
		cudaMemcpyFromSymbol(&duplicates, duplicateCounter, sizeof(int));
		std::cout	<< "\t     Duplicates:  " << duplicates << std::endl << std::endl;
	}
}


//#define fun(a)		BFS_KernelMainGLOB	<BLOCKDIM, (a), false, DUPLICATE_REMOVE>\
//								<<<DIV(FrontierSize, (BLOCKDIM / (a)) * ITEM_PER_WARP), BLOCKDIM, SMem_Per_Block(BLOCKDIM)>>>\
//								(devOutNodes, devOutEdges, colors, devF1, devF2, FrontierSize, level);

#define fun(a)		BFS_KernelMainGLOB	<BLOCKDIM, (a), false, DUPLICATE_REMOVE>\
						<<<gridDim, BLOCKDIM, SMem_Per_Block(BLOCKDIM)>>>\
						(devOutNodes, devOutEdges, colors, devF1, devF2, FrontierSize, level);

#define funB(a)		BFS_KernelMainGLOBB	<BLOCKDIM, (a), false, DUPLICATE_REMOVE>\
						<<<gridDim, BLOCKDIM, SMem_Per_Block(BLOCKDIM)>>>\
						(devOutNodes, devOutEdges, colors, devF1, devF2, FrontierSize, level);

inline void cudaGraph::cudaFASTCON() {
	int SizeArray[4];
	int level = 1, FrontierSize = NUM_SOURCES;
	const int gridDim = (MAX_CONCURR_TH / BLOCKDIM);
	while ( FrontierSize ) {
		// FrontierDebug(FrontierSize, level, PRINT_FRONTIER);
		int size = logValueHost<MIN_VW, MAX_VW>(FrontierSize);

		def_SWITCH(size);

		cudaMemcpyFromSymbolAsync(SizeArray, devF2Size, sizeof(int) * 4);
		FrontierSize = SizeArray[level & 3];
		if (FrontierSize > this->allocFrontierSize)
			error("BFS Frontier too large. Required more GPU memory. N. of Vertices/Edges in frontier: " << FrontierSize << " >  allocated: " << this->allocFrontierSize);

		std::swap<int*>(devF1, devF2);
		level++;
	}
}

#undef fun

void cudaGraph::Reset(const int Sources[], int nof_sources) {
	cudaMemcpy(devF1, Sources, nof_sources * sizeof(int), cudaMemcpyHostToDevice);

	const int ZERO = 0;
	cudaMemcpyToSymbol(devFoundCommon, &ZERO, sizeof(int));

	// Set no-color for every node
	cudaUtil::fillKernel<dist_t><<<DIV(V, 128), 128>>>(colors, V, -1);
	// Color the sources
	cudaUtil::scatterKernel<dist_t><<<DIV(nof_sources, 128), 128>>>(devF1, nof_sources, colors);

	int SizeArray[4] = {0, 0, 0, 0};
	cudaMemcpyToSymbol(devF2Size, SizeArray, sizeof(int) * 4);

	GReset<<<1, 256>>>();
	cudaError("Graph Reset");
}


// ---------------------- AUXILARY FUNCTION ---------------------------------------------

inline void cudaGraph::FrontierDebug(int FrontierSize, int level, bool PRINT_F) {
	totalFrontierNodes += FrontierSize;
	if (PRINT_F == 0)
		return;
	std::stringstream ss;
	ss << "Level: " << level << "\tF2Size: " << FrontierSize << std::endl;
	if (PRINT_F == 2)
		printExt::printCudaArray(devF1, FrontierSize, ss.str());
}

template<int MIN_VALUE, int MAX_VALUE>
inline int cudaGraph::logValueHost(int Value) {
	int logSize = 31 - __builtin_clz(MAX_CONCURR_TH / Value);
	if (logSize < _Log2<MIN_VALUE>::VALUE)
		logSize = _Log2<MIN_VALUE>::VALUE;
	if (logSize > _Log2<MAX_VALUE>::VALUE)
		logSize = _Log2<MAX_VALUE>::VALUE;
	return logSize;
}

		/*if (BLOCK_BFS && FrontierSize < BLOCK_FRONTIER_LIMIT) {
			BFS_BlockKernel<DUPLICATE_REMOVE><<<1, 1024, 49152>>>(devNodes, devEdges, colors, devF1, devF2, FrontierSize);
			cudaMemcpyFromSymbolAsync(&level, devLevel, sizeof(int));
		} else {*/
