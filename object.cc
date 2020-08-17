#include "object.h"
#include "camera.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <immintrin.h>
#include <smmintrin.h>

#define qx(q) q[0]
#define qy(q) q[1]
#define qz(q) q[2]
#define qw(q) q[3]

static size_t GetFileLength(FILE *fp)
{
	fseek(fp, 0, SEEK_END);
	size_t len = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	return len;
}

int LoadTextFile(const char* filename, std::vector<char>& out)
{
	FILE* fp = fopen(filename, "r");
	if(!fp)
	{
		return -1;
	}

	size_t filelen = GetFileLength(fp);
	out.resize(filelen + 1, 0);
	size_t bread = fread(&out[0], 1, filelen, fp);
	fclose(fp);
	return bread;
}

bool CheckShader(GLuint hShader)
{
	GLint success = 0;
	glGetShaderiv(hShader, GL_COMPILE_STATUS, &success);
	return GL_TRUE == success;
}

std::string GetShaderErrorMsg(GLuint hShader)
{
	GLint loglen = 0;
	glGetShaderiv(hShader, GL_INFO_LOG_LENGTH, &loglen);
	char* log = reinterpret_cast<char*>(malloc(loglen + 1));
	glGetShaderInfoLog(hShader, loglen, 0, log);
	std::string retval(log);
	free(log);
	return retval;
}

Object::Object() : m_shader_program(0), m_vbo_reserved(0)
{
	glGenBuffers(1, &m_vbo_vertices);
	glGenVertexArrays(1, &m_vao);
}

Object::~Object()
{
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &m_vbo_vertices);
	glBindVertexArray(0);
	glDeleteVertexArrays(1, &m_vao);

	if(m_shader_program)
	{
		glDeleteProgram(m_shader_program);
		m_shader_program = 0;
	}

}

void Object::AddVertex(const vmath::vec4& v, const vmath::Tvec4<unsigned char>& c)
{
	m_data.emplace_back(Vertex(c, v));
}

void Object::UpdateBuffer()
{
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo_vertices);
	size_t newsize = m_data.size() * sizeof(struct Vertex);
	if(newsize <= m_vbo_reserved) [[likely]]
	{
		glBufferSubData(GL_ARRAY_BUFFER, 0, m_data.size() * sizeof(struct Vertex), &m_data[0]);
	}
	else [[unlikely]]
	{
		m_vbo_reserved = newsize;
		glBufferData(GL_ARRAY_BUFFER, m_vbo_reserved, &m_data[0], GL_DYNAMIC_DRAW);

	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Object::InitBuffer()
{
	glBindVertexArray(m_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo_vertices);

	m_vbo_reserved = m_data.size() * sizeof(struct Vertex);
	glBufferData(GL_ARRAY_BUFFER, m_vbo_reserved, &m_data[0], GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(struct Vertex), (char*)(4));
	glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_FALSE, sizeof(struct Vertex), 0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void Object::Draw(const Camera* pCamera,
		  GLint mode)
{
	glUseProgram(m_shader_program);

	glUniformMatrix4fv(m_model_location, 1, GL_FALSE, m_modeltransform);

	glUniformMatrix4fv(m_view_location, 1, GL_FALSE, pCamera->GetViewTransform());
	glUniformMatrix4fv(m_proj_location, 1, GL_FALSE,
			   pCamera->GetProjectionTransform());

	glBindVertexArray(m_vao);
	glDrawArrays(mode, 0, m_data.size());
	glBindVertexArray(0);

	glUseProgram(0);
}

bool Object::LoadShaders(const char* vertfn, const char* fragfn)
{
	LoadTextFile(vertfn, m_vertshadertext);
	LoadTextFile(fragfn, m_fragshadertext);

	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);

	char* pvshader = &m_vertshadertext[0];
	char* pfshader = &m_fragshadertext[0];

	glShaderSource(vs, 1, &pvshader, 0);
	glShaderSource(fs, 1, &pfshader, 0);
	glCompileShader(vs);
	glCompileShader(fs);

	if(!CheckShader(vs))
	{
		fprintf(stderr, "Shader %d: %s", vs, GetShaderErrorMsg(vs).c_str());
		glDeleteShader(vs);
		glDeleteShader(fs);
		return false;
	}

	if(!CheckShader(fs))
	{
		fprintf(stderr, "Shader %d: %s", fs, GetShaderErrorMsg(fs).c_str());
		glDeleteShader(vs);
		glDeleteShader(fs);
		return false;
	}

	m_shader_program = glCreateProgram();
	glAttachShader(m_shader_program, vs);
	glAttachShader(m_shader_program, fs);
	glLinkProgram(m_shader_program);
	glDeleteShader(fs);
	glDeleteShader(vs);

	m_model_location = glGetUniformLocation(m_shader_program, "modelMat");
	m_proj_location = glGetUniformLocation(m_shader_program, "projMat");
	m_view_location = glGetUniformLocation(m_shader_program, "viewMat");

	return true;
}

 __m128 SSECrossProduct(__m128 vec_a, __m128 vec_b)
{
	__m128 sh_a = _mm_shuffle_ps(vec_a, vec_a, _MM_SHUFFLE(3, 0, 2, 1));
	__m128 sh_b = _mm_shuffle_ps(vec_b, vec_b, _MM_SHUFFLE(3, 0, 2, 1));
	__m128 result = _mm_sub_ps(_mm_mul_ps(vec_a, sh_b), _mm_mul_ps(vec_b, sh_a));
	return _mm_shuffle_ps(result, result, _MM_SHUFFLE(3, 0, 2, 1));
}

void sse_quat_applyrot(const float* a, const float* b, float* out)
{
	//This is substantially faster than SISD
	__m128 vec_b = _mm_load_ps(b);
	__m128 w_a = _mm_load_ps1(&qw(a));
	__m128 proda = _mm_mul_ps(vec_b, w_a);
	__m128 vec_a = _mm_load_ps(a);
	__m128 w_b = _mm_load_ps1(&qw(b));
	__m128 prodb = _mm_mul_ps(vec_a, w_b);
	__m128 result = _mm_add_ps(proda, prodb);
	__m128 xprod = SSECrossProduct(vec_a, vec_b);

	__m128 vec_pq =  _mm_add_ps(result, xprod);
	__m128 w_pq =_mm_sub_ps(_mm_mul_ps(w_a, w_b), _mm_dp_ps(vec_a, vec_b, 0x77));

	__m128 prodpq = _mm_mul_ps(vec_a, w_pq);
	proda = _mm_mul_ps(vec_pq, w_a);
	result = _mm_sub_ps(prodpq, proda);
	xprod = SSECrossProduct(vec_a, vec_pq);
	_mm_store_ps(out, _mm_sub_ps(result, xprod));
	qw(out) = _mm_cvtss_f32(_mm_add_ps(_mm_mul_ps(w_a, w_pq), _mm_dp_ps(vec_a, vec_pq, 0x77)));
}

void sse_quat_mul(const float* a, const float* b, float* out)
{
	//Performance is marginally better than SISD
	__m128 vec_b = _mm_load_ps(b);
	__m128 w_a = _mm_load_ps1(&qw(a));
	__m128 proda = _mm_mul_ps(vec_b, w_a);
	__m128 vec_a = _mm_load_ps(a);
	__m128 w_b = _mm_load_ps1(&qw(b));
	__m128 prodb = _mm_mul_ps(vec_a, w_b);

	__m128 result = _mm_add_ps(proda, prodb);
	__m128 xprod = SSECrossProduct(vec_a, vec_b);
	_mm_store_ps(out, _mm_add_ps(result, xprod));
	float dp = _mm_cvtss_f32(_mm_dp_ps(vec_a, vec_b, 0x77));
	qw(out) = qw(a)*qw(b) - dp;
}

void PrintQuat(float* quat, const char* str)
{
	printf("%10s: %f %f %f %f\n", str, quat[0], quat[1], quat[2], quat[3]);
}

void Object::Rotate(const vmath::Tquaternion<float>& rotation,
		    const vmath::Tquaternion<float>& inverse)
{
	float rot[4], inv[4], r[4];
	memcpy(rot, &rotation[0], 4 * sizeof(float));
	memcpy(inv, &inverse[0], 4 * sizeof(float));

	for(int idx = 0, z = m_data.size(); idx < z; ++idx)
	{
		/*
		m_data[idx].vertex = rotation * m_data[idx].vertex * inverse;
		*/
		float vert[4];
		memcpy(vert, &m_data[idx].vertex[0], 4 * sizeof(float));
		sse_quat_mul(rot, vert, r);
		sse_quat_mul(r, inv, vert);
		memcpy(&m_data[idx].vertex[0], vert, sizeof(float) * 4);

	}
}
