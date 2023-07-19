#include "graph.h"

//init delle variabili del grafo
Graph::Graph(const int _V, const int _E, const GDirection GraphDirection) : V(_V), E(_E), Direction(GraphDirection)
{
	try
	{
		Queue = new int[V]; //vertici da visitare
		Visited.resize(V);	//resize dei vertici visitati

		OutNodes = new int[V + 1]; //off-set dei vertici come in CSR, OutNodes[0] = 0 per definizione
		OutEdges = new int[E];		 //destinazione di ogni edge, OutEdges[0] = destinazione arco 0 e cosi via
		OutDegree = new int[V]();	//out degree dei vertici in queue
		COO_Edges = new int2[E];	 //edges della COO
	}
	catch (std::bad_alloc &exc)
	{
		error("OUT OF MEMORY: Graph too Large (Out Graph)!!");
	}
}

//conversioni da COO a CSR
void Graph::ToCSR()
{
	// std::cout << "        COO To CSR...\t\t" << std::flush;

	for (int i = 0; i < COOSize; i++)
	{
		int index1 = COO_Edges[i].x;
		int index2 = COO_Edges[i].y;
		OutDegree[index1]++;
		if (Direction == UNDIRECTED)
			OutDegree[index2]++;
	}
	OutNodes[0] = 0;
	std::partial_sum(OutDegree, OutDegree + V, OutNodes + 1);

	int *TMP = new int[V]();
	for (int i = 0; i < COOSize; ++i)
	{
		int index1 = COO_Edges[i].x;
		int index2 = COO_Edges[i].y;
		OutEdges[OutNodes[index1] + TMP[index1]++] = index2;
		if (Direction == UNDIRECTED)
			OutEdges[OutNodes[index2] + TMP[index2]++] = index1;
	}
	delete TMP;
	// std::cout << "Complete!\n\n"
	// 					<< std::flush;
}

//graph to COO
void Graph::Dimacs10ToCOO()
{
	std::cout << " Dimacs10th to COO...\t\t" << std::flush;
	int count_Edges = 0;
	for (int i = 0; i < V; i++)
	{
		for (int j = OutNodes[i]; j < OutNodes[i + 1]; j++)
		{
			int dest = OutEdges[j];
			bool flag = true;
			for (int t = OutNodes[dest]; flag && t < OutNodes[dest + 1]; t++)
			{
				if (t == i)
					flag = false;
			}
			if (flag)
				COO_Edges[count_Edges++] = make_int2(i, dest);
		}
	}
	COOSize = count_Edges;
	std::cout << "Complete!\n\n"
						<< std::flush;
}

void Graph::print()
{
	printExt::printArray(COO_Edges, COOSize, "COO Edges\n");
	printExt::printArray(OutNodes, V + 1, "OutNodes\t");
	printExt::printArray(OutEdges, E, "OutEdges\t");
	printExt::printArray(OutDegree, V, "OutDegree\t");
}

//init delle variabili del grafo
void Graph::BfsInit(int source, int *_Distance)
{
	left = 0, right = 1;

	std::fill(Visited.begin(), Visited.end(), false);
	Visited[source] = true;

	Distance = _Distance;
	std::fill(Distance, Distance + V, INF);
	Distance[source] = 0;

	Queue[0] = source;
}

//esecuzione bfs nel grafo
void Graph::bfs(int target)
{
	while (left < right) //finchè ci sono nodi a sinistra continua a prendere indici dalla queue
	{
		int qNode = Queue[left++];

		if (qNode == target)
		{
			std::cout << qNode << "\n";
			std::cout << "sequential connection found\n";
			return;
		}

		for (int i = OutNodes[qNode]; i < OutNodes[qNode + 1]; ++i) //prendo nodi dal vettore degli outnodi
		{
			int dest = OutEdges[i]; //assegno dest al edge di quel nodo

			if (!Visited[dest]) //check se nodo già visitato altrimenti lo visito e sposto la destinazione avanti
			{
				Visited[dest] = true;
				Distance[dest] = Distance[qNode] + 1; //aumento la distanza percorsa per arrivare a dest
				Queue[right++] = dest;
			}
		}
	}
}

//nodi visitati uguale a tutti i nodi a "destra" cioè somma di tutti i last node visitati per queue di nodi
int Graph::visitedNodes()
{
	return right;
}

//edge visitati
int Graph::visitedEdges()
{
	if (right == V) //posso aver visitato tutti i nodi, allora edge visited = total edges
		return E;
	int sum = 0;
	for (int i = 0; i < right; ++i) //se non ho visitato tutti i nodi, allora conto l'outdegree di ogni nodo inserito nella queue fino a right
		sum += OutDegree[Queue[i]];
	return sum;
}

//max distance percorsa da source
int Graph::getMaxDistance()
{
	return Distance[Queue[right - 1]];
}

//frontier usato nella BFS4K, funzionamento su grafo sequenziale
void Graph::bfsFrontier(std::vector<int> &Frontiers)
{
	int oldDistance = 0;
	Frontiers.push_back(1);

	while (left < right)
	{
		int qNode = Queue[left++]; //assegno nodo succesivo dalla queue

		if (Distance[qNode] > oldDistance) //controllo che la distance per quel nodo sia maggiore della distance percorsa ora (praticamente aumento il livello)
		{
			Frontiers.push_back(right - left + 1); //inserisco nel frontier
			oldDistance = Distance[qNode];				 //aggiorno la distanza con la distanza da quel nodo
		}

		for (int i = OutNodes[qNode]; i < OutNodes[qNode + 1]; ++i) //bfs seq
		{
			int dest = OutEdges[i];

			if (!Visited[dest])
			{
				Visited[dest] = true;
				Distance[dest] = Distance[qNode] + 1;
				Queue[right++] = dest;
			}
		}
	}
}

//analisi del degree del grafo, medio, std, min, max
void Graph::DegreeAnalisys()
{
	float avg = (float)E / V;
	float stdDev = fUtil::stdDeviation(OutDegree, V, avg);
	std::pair<int *, int *> minmax = std::minmax_element(OutDegree, OutDegree + V);
	std::cout << std::setprecision(1)
						<< "          Avg:  " << avg << std::endl
						<< "     Std. Dev:  " << stdDev << std::endl
						<< "          Min:  " << *minmax.first << "\t\t" << std::endl
						<< "          Max:  " << *minmax.second << "\t\t" << std::endl;
	std::cout << std::endl;
}
