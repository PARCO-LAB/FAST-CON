#pragma once

long long int totalFrontierNodes;
long long int totalFrontierEdges;
int visitedNodes, FrontierSize, level;
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
	double totalAdjTime = 0;
	double avgRun = 0;

	int topdown = 0;
	int bottomup = 0;
	float doublePartialTime = 0;
	float singleTestTime = 0;
	int globalVisited = 0;
	int found = 0;

	for (int i = 0; i < N_OF_TESTS; i++)
	{
		int *source = new int[NUM_SOURCES]();

		for (int ff = 0; ff < NUM_SOURCES; ++ff)
		{
			source[ff] = N_OF_TESTS == 1 ? 0 : mt::randf_co() * V;
		}

		// totalVisitedNodes += graph.visitedNodes(); //variabili aggiornate per statistiche
		// totalVisitedEdges += graph.visitedEdges();
		// diameter += graph.getMaxDistance();

		// printf("TOP-DOWN PHASE\n");
		phase = 0;
		topdown++;

		bool hostFoundCommon = 0;
		int run = 0;
		globalVisited = 0;

		this->Reset(source, NUM_SOURCES); //reset cuda graph con inserimento del source nel frontier

		while (!hostFoundCommon && FrontierSize)
		{
			visitedNodes = 0;
			// printf("RUN %d\n", ++run);
			++run;

			TM.start();		   //start time
			this->cudaBFS4K(); //start BFS4K
			TM.stop();		   //stop timer

			cudaError("BFS Kernel N");

			//sync device
			// cudaDeviceSynchronize();

			float bfsTime = TM.duration();
			// printf("time of top-down bfs %f\n", bfsTime);

			// int nSources = NUM_SOURCES;
			// cudaMemcpyToSymbolAsync(countTempReachByS, &nSources, sizeof(int));

			// TM.start();
			// AdjAnalysis<<<DIV(MAX_CONCURR_TH, BLOCKDIM), BLOCKDIM>>>(devAdjMatrix);
			// TM.stop();

			int hostCountTempReachByS = 1;
			const int ZERO=0;
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

			cudaMemcpyFromSymbol(&hostFoundCommon, found_common, sizeof(bool));

			// printf("time of top-down adj matrix analysis %f\n", adjTime);
			totalAdjTime += adjTime;
			float partialTime = adjTime + bfsTime;
			totalTopDown += bfsTime;
			singleTestTime += partialTime;
			// printf("partial time top-down %f\n", partialTime);

			// printf("visited nodes %d\n", visitedNodes);
			// printf("level %d\n", level);

			globalVisited += visitedNodes;

			// printf("globalVisiteds nodes %d\n", globalVisited);

			if (hostFoundCommon)
			{
				// printf("FOUND CONNECTION S-T\n");
				found++;
			}

			totalTime += partialTime;
			avgRun += 1;
		}

		// if (N_OF_TESTS > 0) //stampo stats
		// {
		// 	std::cout << "iter: " << std::left << std::setw(10) << i + 1 << "time: " << std::setw(10) << singleTestTime << "visitedNodes: "
		// 			  << std::setw(10) << globalVisited << "Source: " << source[0] << " Target " << source[NUM_SOURCES - 1] << std::endl;
		// 	//std::cout << ss.str() << std::endl;
		// 	std::cout << "-----------------------------------------------------------------------------" << std::endl;
		// }
		// else
		// {
		// 	cudaUtil::Compare(hostDistance, devDistance, V, "Distance Check", 1);
		// }

		singleTestTime = 0;
	}

	std::cout << std::setprecision(2) << std::fixed << std::endl //stats di percorrenza BFS
			  << "\t    Number of TESTS:  " << N_OF_TESTS << std::endl
			  << "\t    Number of SOURCE:  " << NUM_SOURCES << std::endl
			  << "\t    Percentage of visited EDGES:  " << DEEPNESS << std::endl
			  << "\t          Avg. Time:  " << totalTime / N_OF_TESTS << " ms" << std::endl
			  << "\t          Avg. TopDownTime:  " << totalTopDown / topdown << " ms" << std::endl
			  << "\t    Avg. of AdjTime:  " << totalAdjTime / topdown << std::endl
			  << "\t          ST-FOUND:  " << found << std::endl
			  << "\t          Avg. Run:  " << avgRun / N_OF_TESTS << std::endl
			  << std::endl;

	if (COUNT_DUP && N_OF_TESTS == 1)
	{
		int duplicates;
		cudaMemcpyFromSymbol(&duplicates, duplicateCounter, sizeof(int));
		std::cout << "\t     Duplicates:  " << duplicates << std::endl
				  << std::endl;
	}
}

#define fun(a) BFS_KernelMainGLOB<BLOCKDIM, (a), false, DUPLICATE_REMOVE> \
	<<<gridDim, BLOCKDIM, SMem_Per_Block(BLOCKDIM)>>>(devOutNodes, devOutEdges, devDistance, devF1, devF2, FrontierSize, level, devAdjMatrix);

#define funB(a) BFS_KernelMainGLOBB<BLOCKDIM, (a), false, DUPLICATE_REMOVE> \
	<<<gridDim, BLOCKDIM, SMem_Per_Block(BLOCKDIM)>>>(devOutNodes, devOutEdges, devDistance, devF1, devF2, FrontierSize, level, devAdjMatrix);

inline void cudaGraph::cudaBFS4K() //bfs4K primo avvio
{
	while (FrontierSize && visitedNodes <= (graph.E / 100 * DEEPNESS))
	{
		int size = logValueHost<MIN_VW, MAX_VW>(FrontierSize);
		const int gridDim = (MAX_CONCURR_TH / BLOCKDIM);

		// visitedNodes += MIN(MAX_CONCURR_TH / (pow(2, size)), FrontierSize);
		visitedNodes += FrontierSize;

		def_SWITCH(size);

		cudaMemcpyFromSymbolAsync(SizeArray, devF2Size, sizeof(int) * 4);
		FrontierSize = SizeArray[level & 3];

		if (FrontierSize > this->allocFrontierSize)
			error("BFS Frontier too large. Required more GPU memory. N. of Vertices/Edges in frontier: " << FrontierSize << " >  allocated: " << this->allocFrontierSize);

		std::swap<int *>(devF1, devF2);
		level++;
	}
}

#undef fun

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

	//inserimento frontier con sources nel device
	cudaMemcpy(devF1, Sources, nof_sources * sizeof(int), cudaMemcpyHostToDevice);

	//azzeramento found common
	bool hostcommon = 0;
	cudaMemcpyToSymbol(found_common, &hostcommon, sizeof(bool));

	cudaUtil::fillKernel<dist_t><<<DIV(V, 128), 128>>>(devDistance, V, INF);
	cudaUtil::scatterKernel<dist_t><<<DIV(nof_sources, 128), 128>>>(devF1, nof_sources, devDistance, 0); //cudaUtil.cuh

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
	int logSize = 31 - __builtin_clz(MAX_CONCURR_TH / Value); //least significant 0, ultimo 0 prima del primo 1 meno significativo
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