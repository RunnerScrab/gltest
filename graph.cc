#include "graph.h"

Graph::Graph(const std::vector<Vertex>& verts)
{
	m_nodes.reserve(verts.size());
	for(int i = 0, z = verts.size(); i < z; ++i)
	{
		const vmath::vec4& vec = verts[i].vertex;
		m_nodes.emplace_back(
			Node(vmath::vec3(vec[0], vec[1], vec[2]))
			);
	}
}

void Graph::ConnectMST()
{
	//FIXME: A nasty Log(O^2) brute force algorithm
	std::vector<Node*> graph;
	std::vector<Node*> nodes;
	nodes.reserve(m_nodes.size());
	graph.reserve(m_nodes.size());

	for(int i = 0, z = m_nodes.size(); i < z; ++i)
	{
		nodes.push_back(&m_nodes[i]);
	}

	graph.emplace_back(&m_nodes[0]);

	Node* pThis = graph[0];

	while(nodes.size())
	{
		auto it = nodes.begin();
		std::vector<Node*>::iterator pMin = nodes.end();
		float mindist = 0.f;

		for(auto z = nodes.end(); it != z; ++it)
		{
			Node* pNode = *it;
			float dist = pThis->DistanceTo(*pNode);
			if(pMin == nodes.end() || dist < mindist)
			{
				mindist = dist;
				pMin = it;
			}
		}
		graph.push_back(*pMin);
		m_edges.emplace_back(Edge(pThis->GetVert(), (*pMin)->GetVert()));
		pThis = *pMin;
		nodes.erase(pMin);
	}

}
