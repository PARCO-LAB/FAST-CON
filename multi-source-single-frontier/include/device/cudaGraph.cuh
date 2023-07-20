#pragma once

#include <sstream>
#include <iostream>
#include "fUtil.h"
#include "cudaUtil.cuh"
#include "graph.h"
#include "Timer.cuh"
#include "printExt.h"
#include "../../config.h"
#include "definition.cuh"


class cudaGraph {
		Graph graph;
		int* devOutNodes, *devOutEdges, *devOutDegree;
		int *devF1, *devF2;
		dist_t* colors;
		int V, E;
		int allocFrontierSize;
		bool* devAdjMatrix;

	public:
		cudaGraph(Graph& graph);

		void Reset(const int Sources[], int nof_sources = 1);

		inline void cudaFASTCON();
		void cudaFASTCON_N(int nof_tests = 1);

	private:
		inline void FrontierDebug(int F2Size, int level, bool DEBUG = false);

		template<int MIN_VALUE, int MAX_VALUE>
		inline int logValueHost(int Value);
};
