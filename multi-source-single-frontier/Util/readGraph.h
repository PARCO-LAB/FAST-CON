#pragma once

#include <iostream>     // std::cout, std::fixed, peek()
#include <iomanip>      // std::setprecision
#include <string>
#include <fstream>      // std::ifstream
#include <iterator>     // std::istream_iterator
#include <algorithm>
#include <numeric>
#include <sstream>

#include "fUtil.h"	//checkRegularFile(), fileSize(), skipLines(), error()
#include "readUtil.h"
#include "graph.h"
//#include "graphWeight.h"

namespace readGraph {
	void readGraphHeader( const char* File, int &V, int &E, int &nof_lines, GDirection &UserDirection );

	void readSTD( const char* File, Graph& graph, const int nof_lines);
	
	//void readMPG( const char* File, GraphWeight& graph, const int nof_lines);
}
