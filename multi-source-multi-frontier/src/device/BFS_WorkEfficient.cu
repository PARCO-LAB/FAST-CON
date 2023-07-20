#pragma once

long long int totalFrontierNodes = 0;
int level = 1, FrontierSize = NUM_SOURCES;
int steps = 0; // Number of steps executed

void cudaGraph::cudaFASTCON_N(int nof_tests) {
	srand(time(NULL));
	if (DUPLICATE_REMOVE)
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	int* hostDistance = new int[V];
	Timer<DEVICE> TM;

	double totalTime = 0, totalBFSTime = 0, totalAdjMatrixTime = 0;
	double singleTestTime = 0;
	int totalSteps = 0;

	int stConnectionsFound = 0;
	for (int i = 0; i < N_OF_TESTS; i++) {

		// Select random sources (the first is s, the last is t)
		int *sources = new int[NUM_SOURCES];
		for (int j = 0; j < NUM_SOURCES; j++)
			sources[j] = N_OF_TESTS == 1 ? 0 : mt::randf_co() * V;

		bool foundCommon = false;
		steps = 0;
		int visited = 1;
		
		this->Reset(sources, NUM_SOURCES);

		// Loop if the connection has not yet been found and
		// in the previous step some nodes have been visited (i.e. there is at least a non-empty frontier)
		while (!foundCommon && visited)
		{
			std::cout << "Step " << steps << std::endl;

			const int ZERO = 0;
			cudaMemcpyToSymbolAsync(devGlobalVisited, &ZERO, sizeof(int));

			// BFS
			TM.start();
			this->cudaFASTCON();
			TM.stop();
			steps++;

			cudaError("BFS Kernel N");

			double BFSTime = TM.duration();
			std::cout << "Time of BFS visit: " << std::fixed << std::setprecision(2) << BFSTime << " ms\n";

			int hostGlobalLevel = 0;
			int hostActiveBlocks = 0;
			cudaMemcpyFromSymbolAsync(&visited, devGlobalVisited, sizeof(int));

			int hostCountTempReachByS = 1;
			bool allZERO[NUM_SOURCES];
			memset(allZERO, 0, sizeof(bool) * NUM_SOURCES);
			cudaMemcpyToSymbol(checkedSources, allZERO, sizeof(bool) * NUM_SOURCES);

			cudaDeviceSynchronize();
			
			// Adjacency matrix
			TM.start();
			while (hostCountTempReachByS)
			{
				cudaMemcpyToSymbol(countTempReachByS, &ZERO, sizeof(int));
				
				AdjAnalysis<<<NUM_SOURCES, 32>>>(devAdjMatrix);
				cudaDeviceSynchronize();
				
				cudaMemcpyFromSymbol(&hostCountTempReachByS, countTempReachByS, sizeof(int));
			}
			TM.stop();

			cudaDeviceSynchronize();

			double adjTime = TM.duration();

			std::cout << "Time of adjacency matrix analysis: " << std::fixed << std::setprecision(2) << adjTime << " ms\n";

			totalAdjMatrixTime += adjTime;
			totalBFSTime += BFSTime;
			float stepTime = adjTime + BFSTime;
			singleTestTime += stepTime;
			std::cout << "Step time: " << std::fixed << std::setprecision(2) << stepTime << " ms\n";

			cudaMemcpyFromSymbol(&foundCommon, devFoundCommon, sizeof(bool));
			if (foundCommon) {
				std::cout << "Found ST-Connection\n";
				stConnectionsFound++;
			}
		}

		if (nof_tests > 1)
			std::cout 	<< "iter: " << std::left << std::setw(10) << i << "time: " << std::setw(10) << singleTestTime << "source: " << sources[0] << std::setw(10) << " target: " << sources[NUM_SOURCES - 1]
			<< std::endl << "---------------------------------------------------------------------\n";
		else
			cudaUtil::Compare(hostDistance, colors, V, "Distance Check", 1);

		totalSteps += steps;
		totalTime += singleTestTime;
		singleTestTime = 0;
	}

	std::cout	<< std::setprecision(2) << std::fixed << std::endl
				<< "\t             Number of TESTS: " << nof_tests << std::endl
				<< "\t           Number of SOURCES: " << NUM_SOURCES << std::endl
				<< "\t Percentage of visited EDGES: " << DEEPNESS << std::endl
				<< "\t        ST-Connections found: " << stConnectionsFound << std::endl
				<< "\t                   Avg. Time: " << totalTime / nof_tests << " ms" << std::endl
			    << "\t                Avg. BFSTime: " << totalBFSTime / nof_tests << " ms" << std::endl
			    << "\t          Avg. AdjMatrixTime: " << totalAdjMatrixTime / nof_tests << " ms" << std::endl
			    << "\t                    Avg. Run: " << totalSteps / (float)nof_tests << std::endl;

	if (COUNT_DUP && nof_tests == 1) {
		int duplicates;
		cudaMemcpyFromSymbol(&duplicates, duplicateCounter, sizeof(int));
		std::cout	<< "\t     Duplicates:  " << duplicates << std::endl << std::endl;
	}
}

void cudaGraph::cudaFASTCON() {
	BFS_BlockKernel<DUPLICATE_REMOVE><<<NUM_SOURCES, BLOCKDIM>>>(devOutNodes, devOutEdges, colors, devF1, devF2, V, E, devAdjMatrix, steps);
}

void cudaGraph::Reset(const int Sources[], int nof_sources) {
	level = 1;
	FrontierSize = nof_sources;

	cudaMemset(devF1, 0, allocFrontierSize * sizeof(int));
	cudaMemset(devF2, 0, allocFrontierSize * sizeof(int));
	cudaMemset(devAdjMatrix, 0, NUM_SOURCES * NUM_SOURCES * sizeof(bool));

	cudaMemcpy(devF1, Sources, nof_sources * sizeof(int), cudaMemcpyHostToDevice);

	const bool ZERO = 0;
	cudaMemcpyToSymbol(devFoundCommon, &ZERO, sizeof(bool));
	cudaMemcpyToSymbol(devGlobalVisited, &ZERO, sizeof(int));

	// Set no-color for every node
	cudaUtil::fillKernel<dist_t><<<DIV(V, 128), 128>>>(colors, V, -1);
	// Color the sources
	// cudaUtil::scatterKernel<dist_t><<<DIV(nof_sources, 128), 128>>>(devF1, nof_sources, colors);
	cudaUtil::scatterKernel<dist_t><<<NUM_BLOCKS, BLOCKDIM>>>(devF1, nof_sources, colors);

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
