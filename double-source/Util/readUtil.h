#pragma once

#include <string>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <sys/stat.h>
#include <limits>
#include <fstream>      // std::ifstream
#include "fUtil.h"

#if __cplusplus > 199711L || __GXX_EXPERIMENTAL_CXX0X__
	#include <unordered_map>
	class NodeMap: public std::unordered_map<int, int> {
#else
	#include <map>
	class NodeMap: public std::map<int, int> {
#endif	
		public:
			int insertNode(int str);	
	};

	std::string extractFileName(std::string s);
	// --------------------------- I/O ---------------------------------------------------

	namespace readUtil {
		long long int fileSize(const char* File);
		void checkRegularFile(const char* File);
		void skipLines(std::ifstream& fin, const int nof_lines = 1);
	}
	
	void initProgress(int lines);
	void readProgress(int lineProgress);

	struct edge_struct {
		int index1, index2;
		edge_struct(){};
		edge_struct(int _index1, int _index2) : index1(_index1), index2(_index2) {}
	};
