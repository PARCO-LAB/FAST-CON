#include "readUtil.h"

int NodeMap::insertNode(int ID) {
	NodeMap::iterator IT = find(ID);
	if(IT == end()){
		int nodeID = (int) size();
		insert(std::pair<int, int>(ID, nodeID));
		return nodeID;
	}
	return IT->second;
}

/**
	\fn	string extractFileName(char* s)
	\brief	Extract the filename of string s
	
	\param s	string to process
	\return		filename of string s
*/
std::string extractFileName(std::string s) {
	std::string path(s, s.length());
	int found = path.find_last_of(".");
	std::string name2 = path.substr(0, found);

	found = name2.find_last_of("/");
	if (found >= 0)
		name2 = name2.substr(found + 1);

	return name2;
}

// --------------------------- I/O ---------------------------------------------------

namespace readUtil {

	void skipLines(std::ifstream& fin, const int nof_lines) {
		for (int i = 0; i < nof_lines; ++i)
			fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}

	long long int fileSize(const char* File) {
		std::ifstream fin(File);
		fin.seekg (0L, std::ios::beg);
		long long int startPos = fin.tellg();
		fin.seekg (0L, std::ios::end);
		long long int endPos = fin.tellg();
		fin.close();
		return endPos - startPos;
	}


	void checkRegularFile(const char* File) {
		std::ifstream fin(File);
		struct stat sb;
		stat(File, &sb);
		
		if (!fin.is_open() || fin.fail() || (sb.st_mode & S_IFMT) != S_IFREG)
			error( " Error. Read file: " << File )

		fin.close();	
	}

}
// --------------------------- PROGRESS ---------------------------------------------------

int progress, nextChunk;
double fchunk;

void initProgress(int lines) {
	progress = 1;
	fchunk = (double) lines / 100;
	nextChunk = fchunk;
	if (nextChunk < 1)
		std::cout << "100%"  << std::flush;
	else
		std::cout << " 0%"  << std::flush;
}

void readProgress(int lineProgress) {
	if (lineProgress == nextChunk) {
		std::cout << "\b\b\b" << std::setw(2) << progress++ << '%' << std::flush;
		nextChunk = progress * fchunk;
	}
}
