#pragma once

long long int totalFrontierNodes;
long long int totalFrontierEdges;
int visitedNodes, FrontierSize, level, run;
bool phase;
int SizeArray[4];

void cudaGraph::cudaBFS4K_N(int N_OF_TESTS)
{
	srand(time(NULL));	//random
	if (DUPLICATE_REMOVE) //configuro sharedmemory per duplicate removal
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	long long int totalVisitedEdges = 0, totalVisitedNodes = 0, diameter = 0;
	totalFrontierNodes = 0, totalFrontierEdges = 0; //inizializzazione variabili per statistiche BFS

	int *hostDistance = new int[V]; //distanza percorsa sui nodi
	Timer<DEVICE> TM;				//timer

	double totalTime = 0;
	double totalTopDown = 0;
	double totalBottomUp = 0;
	double totalBfs = 0;
	double totalSingleTestTime = 0;
	float timeMin = 1000;
	float timeMax = 0;

	int topdown = 0;
	int bottomup = 0;
	double avgRun = 0;
	float doublePartialTime = 0;
	float adjcheckTime = 0;
	int found = 0;
	bool hostFoundCommon = 0;

	// cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 5000000);

	for (int i = 0; i < N_OF_TESTS; i++)
	{
		int *source = new int[NUM_SOURCES]();

		for (int ff = 0; ff < NUM_SOURCES; ++ff)
		{
			source[ff] = mt::randf_co() * V;
		}

		totalVisitedNodes += graph.visitedNodes(); //variabili aggiornate per statistiche
		totalVisitedEdges += graph.visitedEdges();
		diameter += graph.getMaxDistance();

		// printf("TOP-DOWN PHASE\n");
		phase = 0;
		hostFoundCommon = 0;
		totalSingleTestTime = 0;
		visitedNodes = 0;
		run = 0;
		int hostGlobalVisited = 1;
		topdown++;

		this->Reset(source, NUM_SOURCES); //reset cuda graph con inserimento del source nel frontier

		// while (!hostFoundCommon && visitedNodes < E && hostGlobalVisited)
		while (!hostFoundCommon && hostGlobalVisited)
		{
			// printf("RUN %d\n", run);

			const int ZERO = 0;
			cudaMemcpyToSymbolAsync(devGlobalVisited, &ZERO, sizeof(int));
			cudaMemcpyToSymbolAsync(devGlobalLevel, &ZERO, sizeof(int));
			cudaMemcpyToSymbolAsync(devActiveBlocks, &ZERO, sizeof(int));
			cudaMemcpyToSymbolAsync(countFail, &ZERO, sizeof(int));

			TM.start();		  //start time
			this->BFSBlock(); //start BFS4K
			TM.stop();		  //stop timer

			//sync device
			cudaDeviceSynchronize();

			// cudaError("BFS Kernel N");
			float bfsTime = TM.duration();
			// printf("time of top-down bfs %f\n", bfsTime);

			int hostGlobalLevel = 0;
			int hostActiveBlocks = 0;
			cudaMemcpyFromSymbolAsync(&hostGlobalVisited, devGlobalVisited, sizeof(int));
			cudaMemcpyFromSymbolAsync(&hostGlobalLevel, devGlobalLevel, sizeof(int));
			cudaMemcpyFromSymbolAsync(&hostActiveBlocks, devActiveBlocks, sizeof(int));
			visitedNodes += hostGlobalVisited;
			// printf("count global %d\n", hostGlobalVisited);
			// printf("level global %d\n", hostGlobalLevel);
			// printf("average level %d\n", DIV(hostGlobalLevel, NUM_SOURCES));
			// printf("active blocks %d\n", hostActiveBlocks);

			// int nSources = NUM_SOURCES;
			// cudaMemcpyToSymbol(countTempReachByS, &nSources, sizeof(int));

			int hostCountTempReachByS = 1;
			bool allZERO[NUM_SOURCES];
			memset(allZERO, 0, sizeof(bool) * NUM_SOURCES);
			cudaMemcpyToSymbol(checkedSources, allZERO, sizeof(bool) * NUM_SOURCES);

			// sync device
			cudaDeviceSynchronize();
			
			TM.start();

			while (hostCountTempReachByS)
			{
				cudaMemcpyToSymbol(countTempReachByS, &ZERO, sizeof(int));

				AdjAnalysis<<<NUM_SOURCES, 32>>>(devAdjMatrix);
				cudaDeviceSynchronize();
				
				cudaMemcpyFromSymbol(&hostCountTempReachByS, countTempReachByS, sizeof(int));
			}
			
			TM.stop();

			// sync device
			cudaDeviceSynchronize();

			float adjTime = TM.duration();

			cudaMemcpyFromSymbolAsync(&hostFoundCommon, found_common, sizeof(bool));
			// printf("time of top-down adj matrix analysis %f\n", adjTime);
			adjcheckTime += adjTime;
			float partialTime = adjTime + bfsTime;
			totalSingleTestTime += partialTime;
			if (totalSingleTestTime < timeMin)
				timeMin = totalSingleTestTime;
			if (totalSingleTestTime > timeMax)
				timeMax = totalSingleTestTime;
			totalBfs += bfsTime;
			totalTopDown += partialTime;
			// printf("partial time top-down %f\n", partialTime);
			// printf("partial visitedNodes %d\n", visitedNodes);

			if (hostFoundCommon)
			{
				// printf("FOUND CONNECTION S-T\n");
				found++;
			}

			doublePartialTime = partialTime;
			totalTime += partialTime;

			avgRun += 1;

			run++;
		}

		// if (N_OF_TESTS > 0) //stampo stats
		// {
		// 	std::cout << "iter: " << std::left << std::setw(10) << i + 1 << "time: " << std::setw(10) << totalSingleTestTime << "Edges: "
		// 			  << std::setw(10) << graph.visitedEdges() << "Source: " << source[0] << " Target " << source[NUM_SOURCES - 1] << std::endl;
		// 	//std::cout << ss.str() << std::endl;
		// 	std::cout << "-----------------------------------------------------------------------------" << std::endl;
		// }
	}

	std::cout << std::setprecision(2) << std::fixed << std::endl //stats di percorrenza BFS
			  << "\t    Number of TESTS:  " << N_OF_TESTS << std::endl
			  << "\t    Number of SOURCE:  " << NUM_SOURCES << std::endl
			  << "\t    Percentage of visited EDGES:  " << DEEPNESS << std::endl
			  << "\t          Avg. Time:  " << totalTime / N_OF_TESTS << " ms" << std::endl
			  << "\t          Avg. TopDownTime:  " << totalTopDown / topdown << " ms" << std::endl
			  << "\t          Avg. BFSTime:  " << totalBfs / topdown << " ms" << std::endl
			  << "\t          Avg. adjcheckTime:  " << adjcheckTime / topdown << " ms" << std::endl
			  << "\t          Time MIN:  " << timeMin << " ms" << std::endl
			  << "\t          Time MAX:  " << timeMax << " ms" << std::endl
			  << "\t          Avg. Run:  " << avgRun / N_OF_TESTS << std::endl
			  << "\t          Number of positive ST:  " << found << std::endl
			  << std::endl;

	if (COUNT_DUP && N_OF_TESTS == 1)
	{
		int duplicates;
		cudaMemcpyFromSymbol(&duplicates, duplicateCounter, sizeof(int));
		std::cout << "\t     Duplicates:  " << duplicates << std::endl
				  << std::endl;
	}
}

void cudaGraph::BFSBlock()
{
	BFS_BlockKernel<DUPLICATE_REMOVE><<<NUM_SOURCES, BLOCKDIM>>>(devOutNodes, devOutEdges, devDistance, devF1, devF2, V, E, devAdjMatrix, run);
}

void cudaGraph::Reset(int Sources[], int nof_sources)
{
	//level reset
	level = 1;

	//frontier size reset
	FrontierSize = nof_sources;

	//visited nodes reset
	visitedNodes = 0;

	//reset frontier e matrice adiacenza
	cudaMemset(devF1, 0, allocFrontierSize * sizeof(int));
	cudaMemset(devF2, 0, allocFrontierSize * sizeof(int));
	cudaMemset(devAdjMatrix, 0, NUM_SOURCES * NUM_SOURCES * sizeof(bool));

	//inserimento frontier nel device
	cudaMemcpy(devF1, Sources, nof_sources * sizeof(int), cudaMemcpyHostToDevice);

	//azzeramento found common and counters
	const int ZERO = 0;
	bool hostFoundCommon = 0;
	cudaMemcpyToSymbol(found_common, &hostFoundCommon, sizeof(bool));
	cudaMemcpyToSymbol(devGlobalVisited, &ZERO, sizeof(int));
	cudaMemcpyToSymbol(devGlobalLevel, &ZERO, sizeof(int));
	cudaMemcpyToSymbol(devActiveBlocks, &ZERO, sizeof(int));
	cudaMemcpyToSymbol(countFail, &ZERO, sizeof(int));

	cudaUtil::fillKernel<dist_t><<<DIV(V, 128), 128>>>(devDistance, V, INF);
	cudaUtil::scatterKernel<dist_t><<<NUM_BLOCKS, BLOCKDIM>>>(devF1, NUM_SOURCES, devDistance, 0); //cudaUtil.cuh

	//SizeArray[4] = {0, 0, 0, 0};
	memset(SizeArray, 0, sizeof(SizeArray));
	cudaMemcpyToSymbol(devF2Size, SizeArray, sizeof(int) * 4);

	GReset<<<1, 256>>>();
	cudaError("Graph Reset");
}

// ---------------------- AUXILARY FUNCTION ---------------------------------------------

inline void cudaGraph::FrontierDebug(int FrontierSize, int level, bool PRINT_F)
{
	totalFrontierNodes += FrontierSize;

	if (PRINT_F == 0)
		return;
	std::stringstream ss;
	if (level < 2)
	{
		ss.str("");
		ss << "Level: " << level << "\tF1Size: " << FrontierSize << std::endl;
		printExt::printCudaArray(devF1, FrontierSize, ss.str());
	}

	ss.str("");
	ss << "Level: " << level << "\tF2Size: " << FrontierSize << std::endl;
	printExt::printCudaArray(devF2, FrontierSize, ss.str());
}

template <int MIN_VALUE, int MAX_VALUE>
inline int cudaGraph::logValueHost(int Value)
{
	int logSize = 31 - __builtin_clz(MAX_CONCURR_TH / Value); //conta numero di leading zero da sinistra
	if (logSize < _Log2<MIN_VALUE>::VALUE)
	{
		logSize = _Log2<MIN_VALUE>::VALUE;
	}
	if (logSize > _Log2<MAX_VALUE>::VALUE)
	{
		logSize = _Log2<MAX_VALUE>::VALUE;
	}
	return logSize;
}
