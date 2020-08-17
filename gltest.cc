#include <string>
#include <cstring>
#include <cmath>
#include <cstdio>

#include <algorithm>
#include <vector>

#include <time.h>

#include <GL/glew.h> // include GLEW and new version of GL on Windows
#include <GLFW/glfw3.h> // GLFW helper library

#include <xmmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

#include "vmath.h"
#include "vertex.h"
#include "object.h"
#include "camera.h"
#include "graph.h"

constexpr float PI = 3.14159265358979f;

void AVXVecMatrixMultiply(const float* vec, float(*matrix)[4], float* result)
{
	//This is row-major
	//Load the SIMD registers from RAM
	const __m128 mvec = _mm_load_ps(vec);
	const __m128 row0 = _mm_load_ps(matrix[0]);
	const __m128 row1 = _mm_load_ps(matrix[1]);
	const __m128 row2 = _mm_load_ps(matrix[2]);
	const __m128 row3 = _mm_load_ps(matrix[3]);

	//Duplicate each element of the vector across its own row (SIMD register)
	const __m128 mvec0 = _mm_shuffle_ps(mvec, mvec, _MM_SHUFFLE(0, 0, 0, 0));
	const __m128 mvec1 = _mm_shuffle_ps(mvec, mvec, _MM_SHUFFLE(1, 1, 1, 1));
	const __m128 mvec2 = _mm_shuffle_ps(mvec, mvec, _MM_SHUFFLE(2, 2, 2, 2));
	const __m128 mvec3 = _mm_shuffle_ps(mvec, mvec, _MM_SHUFFLE(3, 3, 3, 3));

	__m128 ires = _mm_set1_ps(0.f);
	ires = _mm_fmadd_ps(mvec0, row0, ires);
	ires = _mm_fmadd_ps(mvec1, row1, ires);
	ires = _mm_fmadd_ps(mvec2, row2, ires);
	ires = _mm_fmadd_ps(mvec3, row3, ires);

	//Load the result back into RAM
	_mm_store_ps(result, ires);
}

void printmat4(GLfloat* mat)
{
	int i = 0;
	for(; i < 16; ++i)
	{
		printf("%f ", mat[i]);
		if(0 == ((i + 1) % 4))
			printf("\n");
	}
	printf("\n");
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	vmath::mat4* p = (vmath::mat4*) glfwGetWindowUserPointer(window);
	Camera* pCam = reinterpret_cast<Camera*>(glfwGetWindowUserPointer(window));

	vmath::vec3& cam_velocity = pCam->GetVelocity();
	float speed = 1.0f, rspeed = 5.f;
	bool bPress = (action == GLFW_PRESS | action == GLFW_REPEAT);

	switch(key)
	{
	case GLFW_KEY_D:
		//cam_velocity[0] = bPress ? speed : 0.f;
		*(pCam->GetRotSpeed()) = bPress ? rspeed : 0.f;
		break;
	case GLFW_KEY_A:
		//cam_velocity[0] = bPress ? -speed : 0.f;
		*(pCam->GetRotSpeed()) = bPress ? -rspeed : 0.f;
		break;
	case GLFW_KEY_W:
		cam_velocity[2] = bPress ? speed : 0.f;
		break;
	case GLFW_KEY_S:
		cam_velocity[2] = bPress ? -speed : 0.f;
		break;
	case GLFW_KEY_E:
		cam_velocity[1] = bPress ? speed : 0.f;
		break;
	case GLFW_KEY_Q:
		cam_velocity[1] = bPress ? -speed : 0.f;
		break;
	default:
		break;
	}
}

class RandGen
{
public:
	RandGen()
	{
		struct timespec ts;
		clock_gettime(CLOCK_REALTIME, &ts);
		m_seed = ts.tv_sec << 32 | ts.tv_nsec;
		srandom(ts.tv_sec | ts.tv_nsec);
	}

	u_int64_t PRNG64()
	{
		//PRNG algo by Vladimir Makarov
		m_wyhash64 += 0x60bee2bee120fc15;
		__uint128_t tmp;
		tmp = (__uint128_t) m_wyhash64 * 0xa3b195354a39b70d;
		u_int64_t m1 = (tmp >> 64) ^ tmp;
		tmp = (__uint128_t)m1 * 0x1b03738712fad5c9;
		u_int64_t m2 = (tmp >> 64) ^ tmp;
		return m2;
	}

	double RandDouble(u_int64_t max)
	{
		return fmod(((double) PRNG64()), ((double) max));
	}

private:
	unsigned long long m_seed, m_wyhash64;
};

float TimeDiffSecs(struct timespec *b, struct timespec *a)
{
	return (b->tv_sec - a->tv_sec) + (b->tv_nsec - a->tv_nsec) / 1000000000.0;
}

void GenerateGrid(Object& obj, int density = 5);

int InitGL(GLFWwindow** ppWindow)
{
	// start GL context and O/S window using the GLFW helper library
	if (!glfwInit())
	{
		fprintf(stderr, "ERROR: could not start GLFW3\n");
		return -1;
	}

	*ppWindow = glfwCreateWindow(1024, 1024, "Projection Test", NULL, NULL);
	if (!*ppWindow)
	{
		fprintf(stderr, "ERROR: could not open window with GLFW3\n");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(*ppWindow);
	// start GLEW extension handler
	glewExperimental = GL_TRUE;
	glewInit();
	// get version info
	const GLubyte* renderer = glGetString(GL_RENDERER); // get renderer string
	const GLubyte* version = glGetString(GL_VERSION); // version as a string
	printf("Renderer: %s\n", renderer);
	printf("OpenGL version supported %s\n", version);

	// tell GL to only draw onto a pixel if the shape is closer to the viewer
	glEnable(GL_DEPTH_TEST); // enable depth-testing
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glfwSwapInterval(1);
	glPointSize(4.f);

	return 0;
}

void ReleaseGL(GLFWwindow* pWindow)
{
	// close GL context and any other GLFW resources
	glfwDestroyWindow(pWindow);
	glfwTerminate();
}

int main()
{
	RandGen randgen;
	GLFWwindow* window = 0;
	if(InitGL(&window) < 0)
	{
		fprintf(stderr, "Initialization failed.\n");
		return 0;
	}

	vmath::mat4 scale = vmath::scale(2.f, 2.f, 2.f);

	struct Vertex axes[6] = {
		{{255, 0, 0, 255}, {-1.f, 0.f, 0.f, 1.f}},
		{{255, 0, 0, 255}, {1.f, 0.f, 0.f, 1.f}},

		{{0, 255, 0, 255}, {0.f, 1.f, 0.f, 1.f}},
		{{0, 255, 0, 255}, {0.f, -1.f, 0.f, 1.f}},

		{{0, 0, 255, 255}, {0.f, 0.f, -1.f, 1.f}},
		{{0, 0, 255, 255}, {0.f, 0.f, 1.f, 1.f}}
	};

	Camera camera;
	camera.SetPosition(vmath::vec3(0.f, 0.f, -2.f));
	camera.LookAt(vmath::vec3(0.f, 0.f, 0.f));

	glfwSetWindowUserPointer(window, &camera);

	Object axesobj, stars_obj, edges_obj;

	edges_obj.SetObjectTransform(vmath::scale(2.f, 2.f, 2.f));
	stars_obj.SetObjectTransform(vmath::scale(2.f, 2.f, 2.f));
	axesobj.SetObjectTransform(vmath::scale(2.f, 2.f, 2.f));

	GenerateGrid(axesobj);

	for(int idx = 0; idx < 6; ++idx)
	{
		axesobj.AddVertex(axes[idx].vertex,
				  axes[idx].color);
	}

	axesobj.InitBuffer();
	axesobj.LoadShaders("axes.vert", "axes.frag");

	float theta = (1.f * PI)/180.f;

	vmath::Tquaternion<float> rotation =
		vmath::Tquaternion<float>(0.f, 1.f * sin(theta/2.f), 0.f, 1.f * cos(theta/2.f));
	vmath::Tquaternion<float> inverse = rotation.inverse();
	//vmath::Tquaternion<float>(0.f, -1.f * sin(theta/2.f), 0.f, 1.f * cos(theta/2.f));
	printf("%f %f %f %f\n", rotation[0], rotation[1], rotation[2], rotation[3]);
	printf("%f %f %f %f\n", inverse[0], inverse[1], inverse[2], inverse[3]);

	for(int idx = 0; idx < 32; ++idx)
	{
		float x = randgen.RandDouble(200)/100.f - 1.f;
		float y = randgen.RandDouble(200)/100.f - 1.f;
		float z = randgen.RandDouble(200)/100.f - 1.f;

		stars_obj.AddVertex(vmath::vec4(x, y, z, 1.f),
				    vmath::Tvec4<unsigned char>(255, 255, 255, 255));
	}

	Graph star_graph(stars_obj.GetVerts());
	star_graph.ConnectMST();

	std::vector<Edge>& edges = star_graph.GetEdges();
	for(int i = 0, z = edges.size(); i < z; ++i)
	{
		Edge& edge = edges[i];
		vmath::vec3& a = edge.v0;
		vmath::vec3& b = edge.v1;
		edges_obj.AddVertex(vmath::vec4(a[0], a[1], a[2], 1.f),
				    vmath::Tvec4<unsigned char>(128, 128, 128, 128));
		edges_obj.AddVertex(vmath::vec4(b[0], b[1], b[2], 1.f),
				    vmath::Tvec4<unsigned char>(128, 128, 128, 128));
	}

	edges_obj.InitBuffer();
	edges_obj.LoadShaders("stars.vert", "stars.frag");

	stars_obj.InitBuffer();
	stars_obj.LoadShaders("stars.vert", "stars.frag");

	glfwSetKeyCallback(window, key_callback);

	struct timespec t_a, t_b;
	memset(&t_a, 0, sizeof(struct timespec));
	memset(&t_b, 0, sizeof(struct timespec));

	float t_del = 0.f;
	float rps = 120.f * (PI/180.f); //radians per second
	char titlebuf[512] = {0};

	while(!glfwWindowShouldClose(window))
	{
		clock_gettime(CLOCK_MONOTONIC, &t_a);
		glViewport( 0, 0, 1024, 1024);

		// wipe the drawing surface clear
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		axesobj.Draw(&camera);
		stars_obj.Draw(&camera, GL_POINTS);
		edges_obj.Draw(&camera, GL_LINES);

		axesobj.Rotate(rotation, inverse);
		stars_obj.Rotate(rotation, inverse);
		edges_obj.Rotate(rotation, inverse);

		axesobj.UpdateBuffer();
		stars_obj.UpdateBuffer();
		edges_obj.UpdateBuffer();

		glfwSwapBuffers(window);
		glfwPollEvents();
		//glfwWaitEvents();
		clock_gettime(CLOCK_MONOTONIC, &t_b);
		t_del = TimeDiffSecs(&t_b, &t_a);

		float dr = (t_del * (*camera.GetRotSpeed()))/2.f;
		rotation[1] = sin(dr); //Adjust rotation amount for time delta
		rotation[3] = cos(dr);
		inverse[1] = -rotation[1];
		inverse[3] = rotation[3];
		camera.Move(camera.GetVelocity() * t_del);
		camera.LookAt(vmath::vec3(0.f, 0.f, 0.f));

		//snprintf(titlebuf, 512, "%f", t_del);
		//glfwSetWindowTitle(window, titlebuf);
	}

	printf("Terminating!\n");

	return 0;
}

void GenerateGrid(Object& obj, int density)
{
	unsigned char alpha = 64;
	vmath::Tvec4<unsigned char> color(0, 0, 64, alpha);

	float offset = 2.f/((float) density);

	for(int yidx = 0; yidx <= density; ++yidx)
	{
		for(int xidx = 0; xidx <= density; ++xidx)
		{
			for(int zidx = 0; zidx <= density; ++zidx)
			{
				float x = -1.f + (xidx * offset);
				float x2 = x;

				float z =  -1.f;
				float z2 = 1.f;

				float y = -1.f + (yidx * offset);
				float y2 = y;

				obj.AddVertex(vmath::vec4(x, y, z, 1.f),
					      color);
				obj.AddVertex(vmath::vec4(x2, y2, z2, 1.f),
					      color);

				x = -1.f + (xidx * offset);
				x2 = x;
				z =  -1.f + (zidx * offset);
				z2 = z;
				y = -1.f;
				y2 = 1.f;

				obj.AddVertex(vmath::vec4(x, y, z, 1.f),
					      color);
				obj.AddVertex(vmath::vec4(x2, y2, z2, 1.f),
					      color);

				x = -1.f;
				x2 = 1.f;
				z =  -1.f + (zidx * offset);
				z2 = z;
				y = -1.f + (yidx * offset);
				y2 = y;

				obj.AddVertex(vmath::vec4(x, y, z, 1.f),
					      color);
				obj.AddVertex(vmath::vec4(x2, y2, z2, 1.f),
					      color);
			}
		}
	}
}
