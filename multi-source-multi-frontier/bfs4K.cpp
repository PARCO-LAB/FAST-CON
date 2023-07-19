#include <iostream>
#include <string>
#include <sstream>

#include "readGraph.h"
#include "Timer.cuh"
#include "graph.h"
#include "cudaUtil.cuh"

#include "cudaGraph.cuh"
#include "config.h"

void Parameters(int argc, char *argv[], GDirection &GDir, int &nof_tests);
void checkConfig();
void requiredMem(int V, int E, int COOSize);

int main(int argc, char *argv[])
{
	// cudaUtil::cudaStatics();

	int V, E, nof_lines;
	GDirection GraphDirection;
	int nof_test = 1;
	Parameters(argc, argv, GraphDirection, nof_test);
	checkConfig();

	readGraph::readGraphHeader(argv[1], V, E, nof_lines, GraphDirection);
	requiredMem(V, E, nof_lines);

	Graph graph(V, E, GraphDirection);
	readGraph::readSTD(argv[1], graph, nof_lines);

	//	graph.print();
	graph.DegreeAnalisys();
	int *hostDistance = new int[V];

	// graph.BfsInit(0, hostDistance);
	// std::vector<int> Frontiers;
	// graph.bfsFrontier(Frontiers);
	// for (int i = 0; i < (int)Frontiers.size(); ++i)
	// {
	// 	std::cout << std::setw(3) << i << ":\t" << Frontiers[i] << std::endl;
	// }

	graph.BfsInit(0, hostDistance);
	
	// std::cout << "Computing Seq. BFS...\n" << std::flush;
	Timer<HOST> TM(1, 23);
	TM.start();

	// graph.bfs();

	// TM.getTime("Host BFS");

	cudaGraph devGraph(graph);

	devGraph.cudaBFS4K_N(N_OF_TESTS);			// 		<-- BFS-4K*/
}

void Parameters(int argc, char *argv[], GDirection &GDir, int &nof_tests)
{
	std::string errString(
			"Syntax Error:\n\n bfs8K <graph_path> [ <graph_direction> ] [ -n <number_of_tests>]\n\n\
        <graph_direction>:\n\
                        -D      force directed graph\n\
                        -U      force undirected graph");

	if (argc < 2)
		error(errString)
				GDir = UNDEFINED;
	for (int i = 2; i < argc; ++i)
	{
		std::string parameter = argv[i];
		if (parameter.compare("-D") == 0)
			GDir = DIRECTED;
		else if (parameter.compare("-U") == 0)
			GDir = UNDIRECTED;
		else if (i + 1 < argc && parameter.compare("-n") == 0 && std::string(argv[i + 1]).find_first_not_of("0123456789") == std::string::npos)
		{
			std::istringstream ss(argv[++i]);
			ss >> nof_tests;
		}
		else
			error(errString)
	}
}

void requiredMem(int V, int E, int COOSize)
{
	int reqMem = ((V + 1) * sizeof(int) + // Vertices
								E * sizeof(int) +				// Edges
								V * sizeof(dist_t))			// Distance
							 / (1024 * 1024);


	//printf("V %d, E %d distance %d\n", (V + 1) * sizeof(int), E * sizeof(int), V * sizeof(dist_t));

	// std::cout << "Requested Device Memory: " << reqMem << " MB\n\n";
}

void checkConfig()
{
	if (COUNT_DUP && N_OF_TESTS != 1)
		error("Config Error:  Count Duplicate only with N_OF_TESTS == 1");
	if (INTER_BLOCK_SYNC && DYNAMIC_PARALLELISM)
		error("Config Error:  INTER_BLOCK_SYNC <-> DYNAMIC_PARALLELISM");
	if (INF != (dist_t)-1)
		error("Config Error:  dist_t Type: use only unsigned type");
	if (BLOCKDIM < 256 && INTER_BLOCK_SYNC)
		error("Config Error:  BLOCKDIM < 256  with Inter-block Synchronization");
	if (SAFE && INTER_BLOCK_SYNC)
		std::cout << "\nSAFE <-> INTER_BLOCK_SYNC ...strange...\n";
	if (!SAFE && DYNAMIC_PARALLELISM)
		std::cout << "\nSAFE <-> DYNAMIC_PARALLELISM ...Dynamic parallelism is not active with SAFE == 0..\n";

	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	int max_threads = devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor;
	if (MAX_CONCURR_TH != max_threads)
		error("\nConfig Error:  MAX_CONCURR_TH in config.h != \"Max Resident Thread\"\n");
}
