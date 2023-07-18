#include "readGraph.h"

// -------------- PRIVATE HEADER -------------------------------

void getMatrixMarketHeader(std::ifstream& fin, int &V, int &nof_lines, GDirection &FileDirection);
void getDimacs9Header(std::ifstream& fin, int &V, int &nof_lines);
void getDimacs10Header(std::ifstream& fin, int &V, int &nof_lines, GDirection &FileDirection);
void getSnapHeader(std::ifstream& fin, int &V, int &nof_lines, GDirection &FileDirection);
void getPGSolverHeader(std::ifstream& fin, int &V, int &nof_lines, GDirection &FileDirection);

// -------------- IMPLEMENTATION -------------------------------
namespace readGraph {

void readGraphHeader(const char* File, int &V, int &E, int &nof_lines, GDirection &UserDirection) {
	readUtil::checkRegularFile(File);
	long long int size = readUtil::fileSize(File);
	std::cout.imbue(std::locale(std::locale(), new fUtil::myseps));

	std::string name(File);
	std::cout << "\nRead Header:\t" << name.substr(name.find_last_of("/") + 1) << "\tSize: " << size / (1024 * 1024) << " MB\n";
	std::ifstream fin(File);
	std::string s;
	fin >> s;
	fin.seekg(std::ios::beg);

	GDirection FileDirection = UNDEFINED;
	if (s.compare("c") == 0 || s.compare("p") == 0)
		getDimacs9Header(fin, V, nof_lines);
	else if (s.compare("##") == 0)
		getPGSolverHeader(fin, V, nof_lines, FileDirection);
	else if (s.compare("%%MatrixMarket") == 0)
		getMatrixMarketHeader(fin, V, nof_lines, FileDirection);
	else if (s.compare("#") == 0)
		getSnapHeader(fin, V, nof_lines, FileDirection);
	else if (s.compare("%") == 0 || std::find_if( s.begin(), s.end(), std::not1( std::ptr_fun(::isdigit) )) == s.end() )
		getDimacs10Header(fin, V, nof_lines, FileDirection);
	else
		error( " Error. Graph Type not recognized: " << File << " " << s)

	fin.close();
	bool undirectedFlag = UserDirection == UNDIRECTED || (UserDirection == UNDEFINED && FileDirection == UNDIRECTED);

	E = undirectedFlag ? nof_lines * 2 : nof_lines;
	std::string graphDir = undirectedFlag ? "GraphType: Undirected " : "GraphType: Directed ";

	if (UserDirection != UNDEFINED)
		graphDir.append("(User Def)");
	else if (FileDirection != UNDEFINED) {
		graphDir.append("(File Def)");
		UserDirection = FileDirection;
	} else {
		graphDir.append("(UnDef)");
		UserDirection = DIRECTED;
	}

	std::cout	<< "\n\tNodes: " << V << "\tEdges: " << E << '\t' << graphDir
				<< "\tDegree AVG: " << std::fixed << std::setprecision(1) << (float) E  / V << std::endl << std::endl;
	std::cout.imbue(std::locale());
}

}


void getPGSolverHeader(std::ifstream& fin, int &V, int &nof_lines, GDirection &FileDirection) {
	readUtil::skipLines(fin);
	FileDirection = DIRECTED;

	int p0, p1;
	fin >> p0 >> p1 >> nof_lines;
	V = p0 + p1;
}

void getMatrixMarketHeader(std::ifstream& fin, int &V, int &nof_lines, GDirection &FileDirection) {
	std::string MMline;
	std::getline(fin, MMline);
	FileDirection = MMline.find("symmetric") != std::string::npos ? UNDIRECTED : DIRECTED;
	while (fin.peek() == '%')
		readUtil::skipLines(fin);

	fin >> V >> MMline >> nof_lines;
}

void getDimacs9Header(std::ifstream& fin, int &V, int &nof_lines) {
	while (fin.peek() == 'c')
		readUtil::skipLines(fin);

	std::string nil;
	fin >> nil >> nil >> V >> nof_lines;
}

void getDimacs10Header(std::ifstream& fin, int &V, int &nof_lines, GDirection &FileDirection) {
	while (fin.peek() == '%')
		readUtil::skipLines(fin);

	std::string str;
	fin >> V >> nof_lines >> str;
	FileDirection = str.compare("100") == 0 ? DIRECTED : UNDIRECTED;
}

void getSnapHeader(std::ifstream& fin, int &V, int &nof_lines, GDirection &FileDirection) {
	std::string tmp;
	fin >> tmp >> tmp;
	FileDirection = tmp.compare("Undirected") == 0 ? UNDIRECTED : DIRECTED;
	readUtil::skipLines(fin);

	while (fin.peek() == '#') {
		std::getline(fin, tmp);
		if (tmp.substr(2, 6).compare("Nodes:") == 0) {
			std::istringstream stream(tmp);
			stream >> tmp >> tmp >> V >> tmp >> nof_lines;
			return;
		}
	}
}
