#include "../../include/host/readGraph.h"

// -------------- PRIVATE HEADER -------------------------------

void readDegreeMatrixMarket( std::ifstream& fin, int* Degree, const int nof_lines, GDirection GraphDirection );
void readDegreeDimacs9( std::ifstream& fin, int* Degree, const int nof_lines, GDirection GraphDirection );
void readDegreeDimacs10( std::ifstream& fin, int* Degree, const int V );
void readDegreeSnap( std::ifstream& fin, int* Degree, const int nof_lines, GDirection GraphDirection );

// -------------- IMPLEMENTATION -------------------------------

void readGraphDegree( const char* File, int*& Degree, const int V, const int nof_lines, GDirection GraphDirection) {
	std::cout << "Reading Graph File..." << std::flush;

	Degree = new int[V]();
	if (Degree == NULL)
		error("OUT OF MEMORY: Graph too Large!!");

	std::ifstream fin(File);
	std::string s;
	fin >> s;
	fin.seekg(std::ios::beg);

	//MatrixMarket
	if (s.compare("%%MatrixMarket") == 0)
		readDegreeMatrixMarket(fin, Degree, nof_lines, GraphDirection);
	//Dimacs10
	else if (s.compare("%") == 0 || std::find_if( s.begin(), s.end(), std::not1( std::ptr_fun(::isdigit) )) == s.end() )
		readDegreeDimacs10(fin, Degree, V);
	//Dimacs9
	else if (s.compare("c") == 0 || s.compare("p") == 0)
		readDegreeDimacs9(fin, Degree, nof_lines, GraphDirection);
	//SNAP
	else if (s.compare("#") == 0)
		readDegreeSnap(fin, Degree, nof_lines, GraphDirection);

	fin.close();
	std::cout << "\tComplete!" << std::endl << std::endl << std::flush;

	fUtil::printArray(Degree, V, "Degree\t", DEBUGREAD);
	std::cout.imbue(std::locale());
}


void readDegreeMatrixMarket( std::ifstream& fin, int* Degree, const int nof_lines, GDirection GraphDirection ) {
	initProgress(nof_lines);

	while (fin.peek() == '%')
		readUtil::skipLines(fin);
	readUtil::skipLines(fin);

	for (int lines = 0; lines < nof_lines; ++lines) {
		int index1, index2;
		fin >> index1 >> index2;
		readUtil::skipLines(fin);
		index1--;
		index2--;

		Degree[index1]++;
		if (GraphDirection == UNDIRECTED)
			Degree[index2]++;
		readProgress(lines + 1);
	}
}

void readDegreeDimacs9(	std::ifstream& fin, int* Degree, const int nof_lines, GDirection GraphDirection ) {
	initProgress(nof_lines);

	char c = fin.peek();
	int lines = 0;
	std::string nil;
	while (c != EOF) {

		if (c == 'a') {
			int index1, index2;
			fin >> nil >> index1 >> index2;
			index1--;
			index2--;

			Degree[index1]++;
			if (GraphDirection == UNDIRECTED)
				Degree[index2]++;
			lines++;
			readProgress(lines);
		}
		readUtil::skipLines(fin);
		c = fin.peek();
	}
}

void readDegreeDimacs10( std::ifstream& fin, int* Degree, const int V ) {
	initProgress(V);

	while (fin.peek() == '%')
		readUtil::skipLines(fin);
	readUtil::skipLines(fin);

	for (int lines = 0; lines < V; lines++) {
		std::string str;
		std::getline(fin, str);

		std::istringstream stream(str);
		std::istream_iterator<std::string> iis(stream >> std::ws);

		Degree[lines] = std::distance(iis, std::istream_iterator<std::string>());
		readProgress(lines + 1);
	}
}

void readDegreeSnap(	std::ifstream& fin, int* Degree, const int nof_lines, GDirection GraphDirection ) {
	initProgress(nof_lines);

	while (fin.peek() == '#')
		readUtil::skipLines(fin);

	NodeMap Map;
	for (int lines = 0; lines < nof_lines; lines++) {
		int ID1, ID2;
		fin >> ID1 >> ID2;

		int index1 = Map.insertNode(ID1);
		int index2 = Map.insertNode(ID2);

		Degree[index1]++;
		if (GraphDirection == UNDIRECTED)
			Degree[index2]++;
		readProgress(lines + 1);
	}
}
