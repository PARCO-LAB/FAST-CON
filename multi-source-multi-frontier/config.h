#pragma once

#include <limits>

const int N_OF_TESTS = 100; // NUMBER OF TESTS

//MULTI SORGENTE - MULTI FRONTIERA - RTX2070
#define NUM_SM 16
#define MAX_CONCURR_TH (NUM_SM * 2048) // Number of Resident Threads
#define BLOCKDIM 512
#define NUM_BLOCKS MAX_CONCURR_TH/BLOCKDIM
#define SOURCE_PER_BLOCK 1
#define NUM_SOURCES MAX_CONCURR_TH/BLOCKDIM
#define DEEPNESS 3
#define SHAREDMEMORY 1000000

// -------------------------------------------------------------------------------------
typedef unsigned short dist_t; // ONLY UNSIGNED TYPE	(suggested unsigned short for high-diameter graph, unsigned char for low-diamter graph)

#define MIN_VW 2  // Min Virtual Warp
#define MAX_VW 32 // Max Virtual Warp (suggested: 32)

const int REG_QUEUE = 16;
#define SAFE 1// Check for overflow in REG_QUEUE	(suggested: 1 for high-degree graph, 0 otherwise)

const bool DUPLICATE_REMOVE = 0; // Remove duplicate vertices in the frontier
const bool INTER_BLOCK_SYNC = 0; // Inter-block Synchronization	(suggested: 1 for high-diameter, 0 otherwise)

#define STORE_MODE (FrontierWrite::SIMPLE) // SIMPLE, SHARED_WARP, SHARED_BLOCK

// ------------------------------ DYNAMIC PARALLELISM -----------------------------------

const bool DYNAMIC_PARALLELISM = 0; //(suggested: 1 for high-degree graph, 0 otherwise)
const int THRESHOLD_G = 500000;      // degree threshold to active DYNAMIC_PARALLELISM
const int RESERVED_BLOCKS = 2;
const int LAUNCHED_BLOCKS = 2;

// ----------------------------- ADVANCED CONFIGURATION -----------------------

#define ATOMICCAS 0

const int ITEM_PER_WARP = 1; // Iteration per Warp in the Frontier

const bool BLOCK_BFS = 0;
const int BLOCK_FRONTIER_LIMIT = 2031; //2031 max

// ---------------------------- DEBUG and help constant ---------------------------------------

const bool COUNT_DUP = 0; // count the number of duplicates found with the hash table

const bool CHECK_TRAVERSED_EDGES = false;
const bool PRINT_FRONTIER = false;

const dist_t INF = std::numeric_limits<dist_t>::max();
