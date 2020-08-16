#version 400

in vec4 color;
out vec4 frag_colour;

void main()
{
	if(color.a < 0.1)
	{
		discard;
	}

	frag_colour = color;
};
