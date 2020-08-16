#version 400

layout(location = 0) in vec4 vp;
layout(location = 1) in vec4 incol;
uniform mat4x4 projMat;
uniform mat4x4 viewMat;

uniform mat4x4 modelMat;
uniform mat4x4 scaleMat;
out vec4 color;

float InvSquare(vec3 v)
{
	return 1.0 / max(pow(v.x, 2.0) + pow(v.y, 2.0) + pow(v.z, 2.0), 1.0);
}

void main()
{
	gl_Position =  projMat *viewMat * modelMat * vp;
	float str = InvSquare(vec3(gl_Position.x, gl_Position.y, gl_Position.z)) + 0.4;
	gl_PointSize = 1.0 + (str * 3.0);
	color = vec4((incol[0]/255.0),
		(incol[1]/255.0),
		(incol[2]/255.0),
		(incol[3]/255.0));
};
