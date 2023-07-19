#pragma once

#include <vector_types.h>	//int2
#include <limits>
#include <vector>
#include <algorithm>
#include "fUtil.h"
#include "printExt.h"
#include <utility>
#include "../../config.h"

enum GDirection { DIRECTED = 0, UNDIRECTED = 1, UNDEFINED = 2 };
enum 	  GType { UNDEFTYPE, GRAPH, MULTIGRAPH };

class Graph {
	private:
		int left, right;
		int* Queue;
		std::vector<bool> Visited;
		int *Distance;

	public:
		int2* COO_Edges;
		int* OutNodes, *OutEdges, *OutDegree;
		int V, E, COOSize;
		GDirection Direction;

		Graph(const int _V, const int _E, const GDirection _Direction);

		void ToCSR();
		void print();
		void Dimacs10ToCOO();

		void BfsInit(int source, int* _Distance);
		void bfs(int target);
		void bfsFrontier(std::vector<int>& Frontiers);
		int visitedNodes();
		int visitedEdges();
		int getMaxDistance();

		void DegreeAnalisys();
};
