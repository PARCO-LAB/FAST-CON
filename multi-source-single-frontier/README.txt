--------------------------------------------
REQUIREMENTS
--------------------------------------------

cmake version >= 3.5
cuda version >= 7
NVIDIA Kepler GPU

--------------------------------------------
COMPILE
--------------------------------------------

$ cd build
$ cmake -DARCH=<your_compute_cabability> ..
$ make

example:
$ cmake -DARCH=35 ..

--------------------------------------------
USAGE
--------------------------------------------

$ bfs4K <graph_path> [ -D | -U ]

-D		Force Directed Graph
-U		Force Undirected Graph
without the optional parameter is used the property Directed/Undirected specified in the input file (suggested)

config.h	-> configure BFS-4K

--------------------------------------------
SUPPORTED INPUT FORMAT
--------------------------------------------

SNAP, METIS, GTGRAPH, MATRIX MARKET (mtx), DIMACS9TH, DIMACS10TH
