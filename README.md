# FAST-CON
This repository contains the FAST-CON implementations (double-source, multi-source-single-frontier and multi-source-multi-frontier).
The code is based on BFS-4K (https://profs.scienze.univr.it/~bombieri/BFS-4K/).

## Motivations and contributions
The decision problem of ST-CON in a graph aims to determine whether vertex _t_ can be reached from vertex _s_. Various parallel solutions for GPUs have been proposed in existing literature to address this problem. However, the most efficient approach, relying on two concurrent BFS starting from vertices _s_ and _t_, exhibits limitations when applied to sparse graphs with a low average degree.

FAST-CON ([multi-source-multi-frontier](https://github.com/PARCO-LAB/FAST-CON/tree/main/multi-source-multi-frontier)), implements different strategies to leverage the parallelism offered by GPU architectures on sparse graphs.

The following features are implemented to obtain better performance:
- Multi-Source approach to concurrently visit many nodes
- Step-by-step visit to stop the BFS after a predefined number of visited edges to check if the adjacency matrix contains a connection between _s_ and _t_
- Adjacency matrix to keep track of the connections between multiple sources
- Block independent BFS and frontiers to minimize the number of synchronizations
- Persistent threads to reduce the overhead of the kernel calls

![MSMFSTCON](https://github.com/PARCO-LAB/FAST-CON/assets/32203200/46386afd-c5fd-4208-b83c-1ef71118fce1)

## How to run
Inside every folder there is a README.txt that explains the commands to compile and run.

## Comparison
The [multi-source-multi-frontier](https://github.com/PARCO-LAB/FAST-CON/tree/main/multi-source-multi-frontier) code also does a comparison with a sequential implementation and shows the results: the sequential implementation ranges approximately from ~3.0ms to ~14.0ms, while FAST-CON takes around 0.70ms (time varies based on the selected source and target) on the [road-luxembourg-osm](https://github.com/PARCO-LAB/FAST-CON/blob/main/example_graphs/road-luxembourg-osm.mtx) graph.
