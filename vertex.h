#ifndef VERTEX_H_
#define VERTEX_H_
#include "vmath.h"

struct Vertex
{
	vmath::Tvec4<unsigned char> color;
	vmath::vec4 vertex;

	Vertex(const vmath::Tvec4<unsigned char>& c,
	       const vmath::vec4& v) : color(c), vertex(v)
	{

	}
};

#endif
