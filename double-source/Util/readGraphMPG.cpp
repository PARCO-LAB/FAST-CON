#include "readGraph.h"

// -------------- PRIVATE HEADER -------------------------------

void readGraphPGSolver(	std::ifstream& fin, GraphWeight& graph, const int nof_lines, int& PlayerTH );

// -------------- IMPLEMENTATION -------------------------------

namespace readGraph {

void readMPG( const char* File, GraphWeight& graph, const int nof_lines) {
	std::cout << "Reading Graph File..." << std::flush;

	std::ifstream fin(File);
	std::string s;
	fin >> s;
	fin.seekg(std::ios::beg);

	int PlayerTH;
	readGraphPGSolver(fin, graph, nof_lines, PlayerTH);
	
	fin.close();
	std::cout << "\tComplete!\n" << std::flush;

	graph.ToCSR();
	graph.setPlayerTH(PlayerTH);
}

}	//end namespace


void readGraphPGSolver(	std::ifstream& fin, GraphWeight& graph, const int nof_lines, int& PlayerTH) {
	initProgress(nof_lines);

	readUtil::skipLines(fin);
	fin >> PlayerTH;
	readUtil::skipLines(fin, 2);

	for (int lines = 0; lines < nof_lines; ++lines) {
		int index1, index2, weight;
		fin >> index1 >> index2 >> weight;
		
		graph.COO_Edges[lines] = make_int3(index1, index2, weight);

		readProgress(lines + 1);
		readUtil::skipLines(fin);
	}
	graph.COOSize = nof_lines;
}
