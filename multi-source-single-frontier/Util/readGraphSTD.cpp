#include "readGraph.h"

// -------------- PRIVATE HEADER -------------------------------

void readGraphMatrixMarket(	std::ifstream& fin, Graph& graph, const int nof_lines );
void readGraphDimacs9(	std::ifstream& fin, Graph& graph, const int nof_lines );
void readGraphDimacs10(	std::ifstream& fin, Graph& graph );
void readGraphSnap(	std::ifstream& fin, Graph& graph, const int nof_lines );

// -------------- IMPLEMENTATION -------------------------------

namespace readGraph {

void readSTD( const char* File, Graph& graph, const int nof_lines) {
	std::cout << "Reading Graph File..." << std::flush;

	std::ifstream fin(File);
	std::string s;
	fin >> s;
	fin.seekg(std::ios::beg);

	int dimacs10 = false;
	//MatrixMarket
	if (s.compare("%%MatrixMarket") == 0)
		readGraphMatrixMarket(fin, graph, nof_lines);
	//Dimacs10
	else if (s.compare("%") == 0 || std::find_if( s.begin(), s.end(), std::not1( std::ptr_fun(::isdigit) )) == s.end() ) {
		readGraphDimacs10(fin, graph);
		dimacs10 = true;
	}
	//Dimacs9
	else if (s.compare("c") == 0 || s.compare("p") == 0)
		readGraphDimacs9(fin, graph, nof_lines);
	//SNAP
	else if (s.compare("#") == 0)
		readGraphSnap(fin, graph, nof_lines);

	fin.close();
	std::cout << "\tComplete!\n" << std::flush;

	if (dimacs10 && graph.Direction == UNDIRECTED )
		graph.Direction = UNDEFINED;
	
	graph.ToCSR();
		
	if (dimacs10 && graph.Direction == UNDEFINED ) {
		graph.Dimacs10ToCOO();
		graph.Direction = UNDIRECTED;
	}
}

}	//end namespace


void readGraphMatrixMarket(	std::ifstream& fin, Graph& graph, const int nof_lines) {
	initProgress(nof_lines);

	while (fin.peek() == '%')
		readUtil::skipLines(fin);
	readUtil::skipLines(fin);

	for (int lines = 0; lines < nof_lines; ++lines) {
		int index1, index2;
		fin >> index1 >> index2;
		index1--;
		index2--;

		graph.COO_Edges[lines] = make_int2(index1, index2);

		readProgress(lines + 1);
		readUtil::skipLines(fin);
	}
	graph.COOSize = nof_lines;
}


void readGraphDimacs9(	std::ifstream& fin, Graph& graph, const int nof_lines) {
	initProgress(nof_lines);

	char c;
	int lines = 0;
	std::string nil;
	while ((c = fin.peek()) != EOF) {
		if (c == 'a') {
			int index1, index2;
			fin >> nil >> index1 >> index2;
			index1--;
			index2--;

			graph.COO_Edges[lines] = make_int2(index1, index2);

			lines++;
			readProgress(lines);
		}
		readUtil::skipLines(fin);
	}
	graph.COOSize = lines;
}

void readGraphDimacs10(	std::ifstream& fin, Graph& graph ) {
	initProgress(graph.V);
	while (fin.peek() == '%')
		readUtil::skipLines(fin);
	readUtil::skipLines(fin);

	int countEdges = 0;
	for (int lines = 0; lines < graph.V; lines++) {
		std::string str;
		std::getline(fin, str);

		std::istringstream stream(str);
		std::istream_iterator<std::string> iis(stream >> std::ws);

		int degree = std::distance(iis, std::istream_iterator<std::string>());

		std::istringstream stream2(str);
		for (int j = 0; j < degree; j++) {
			int index2;
			stream2 >> index2;
			
			graph.COO_Edges[countEdges++] = make_int2(lines, index2 - 1);
		}
		readProgress(lines + 1);
	}
	graph.COOSize = countEdges;
}

void readGraphSnap(	std::ifstream& fin, Graph& graph, const int nof_lines ) {
	initProgress(nof_lines);
	while (fin.peek() == '#')
		readUtil::skipLines(fin);

	NodeMap Map;
	for (int lines = 0; lines < nof_lines; lines++) {
		int ID1, ID2;
		fin >> ID1 >> ID2;
		int index1 = Map.insertNode(ID1);
		int index2 = Map.insertNode(ID2);

		graph.COO_Edges[lines] = make_int2(index1, index2);

		readProgress(lines + 1);
	}
	graph.COOSize = nof_lines;
}
