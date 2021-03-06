#version 400

layout(location = 0) in vec4 vp;
layout(location = 1) in vec4 incol;

uniform mat4x4 projviewmodelMat;
uniform mat4x4 projMat;
uniform mat4x4 viewMat;

uniform mat4x4 modelMat;

out vec4 color;

float InvSquare(vec3 v)
{
	return 1.0 / (pow(v.x, 2.0) + pow(v.y, 2.0) + pow(v.z, 2.0));
}

void main()
{
	gl_Position =  projviewmodelMat * vp;
	float str = 1.f;

	color = vec4((incol[0]/255.0) * str,
		(incol[1]/255.0) * str,
		(incol[2]/255.0) * str,
		(incol[3]/255.0) * str);
};
