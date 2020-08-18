#ifndef OBJECT_H_
#define OBJECT_H_
#include <vector>

#include <GL/glew.h> // include GLEW and new version of GL on Windows
#include <GLFW/glfw3.h> // GLFW helper library
#include "vmath.h"
#include "object.h"
#include "vertex.h"

class Camera;

class Object
{
public:
	Object(GLuint drawmode = GL_LINES);
	~Object();

	void AddVertex(const vmath::vec4& v, const vmath::Tvec4<unsigned char>& c);
	void UpdateBuffer();
	void InitBuffer();
	bool LoadShaders(const char* vertfn, const char* fragfn);

	void SetObjectTransform(const vmath::mat4& transform)
	{
		//Scale, position, rotation of object
		m_modeltransform = transform;
	}

	void Rotate(const vmath::Tquaternion<float>& rotation,
		const vmath::Tquaternion<float>& inverse);
	void Move(const vmath::vec3& offset);
	void ClearVerts()
	{
		m_data.clear();
	}

	std::vector<Vertex>& GetVerts()
	{
		return m_data;
	}

	void Draw(const Camera* pCamera);

	GLuint GetShaderProgram() const
	{
		return m_shader_program;
	}
private:

	std::vector<char> m_vertshadertext, m_fragshadertext;
	std::vector<Vertex> m_data;
	vmath::mat4 m_modeltransform;
	GLuint m_vbo_vertices;
	GLuint m_vao;
	GLuint m_shader_program;
	GLuint m_model_location, m_proj_location, m_view_location;
	GLuint m_combined_location;
	GLuint m_drawmode;

	size_t m_vbo_reserved;
};


#endif
