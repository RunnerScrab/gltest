#ifndef GRAPH_H_
#define GRAPH_H_
#include <vector>
#include "vertex.h"
#include "vmath.h"

class Node
{
public:
	Node(const vmath::vec3& v)
	{
		vert = v;
	}

	float DistanceTo(const Node& other)
	{
		return vmath::distance(vert, other.vert);
	}

	vmath::vec3& GetVert()
	{
		return vert;
	}
private:
	vmath::vec3 vert;
};

struct Edge
{
	vmath::vec3 v0, v1;
	Edge(const vmath::vec3& a, const vmath::vec3& b)
	{
		v0 = a;
		v1 = b;
	}
};

class Graph
{
public:
	Graph(const std::vector<Vertex>& verts);

	void ConnectMST();

	std::vector<Edge>& GetEdges()
	{
		return m_edges;
	}
private:
	std::vector<Edge> m_edges;
	std::vector<Node> m_nodes;
};

#endif
