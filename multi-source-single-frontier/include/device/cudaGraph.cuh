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

//classe del cudagraph
class cudaGraph {
		Graph graph;
		int* devOutNodes, *devOutEdges, *devOutDegree; //degree medio, outnodes, outedges
		int *devF1, *devF2; //frontier f1 (totale senza correction), frontier f2 (parziale con correction)
		dist_t* devDistance; //distance del grafo
		int V, E; //archi e nodi del grafo
		int allocFrontierSize; //allocazione della grandezza del frontier su GPU
		bool *devAdjMatrix; //matrice di adiacenza

	public:
		cudaGraph(Graph& graph); //creazione del grafo

		void Reset(int Sources[], int nof_sources = 1); //reset per impostare source nel frontier (di base 1, modificabile per N)

		inline void cudaBFS4K(); //chiamata a BFS4K
		void cudaBFS4K_N(int nof_tests = 1); //call a BFS4K, N test distinti con N source randomiche generate run-time

	private:
		inline void FrontierDebug(int F2Size, int level, bool DEBUG = false); //debug del frontier F2, pre-correction e swap in F2

		template<int MIN_VALUE, int MAX_VALUE>
		inline int logValueHost(int Value);
};
