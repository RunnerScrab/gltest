#include <string>
#include <cstring>
#include <cmath>
#include <cstdio>

#include <algorithm>
#include <vector>

#include <time.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <xmmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

#include "ssemath.h"
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

struct Simulation
{
	Camera* pCamera;
	Object* pStarsObj;
};

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	struct Simulation* pSim =
		reinterpret_cast<struct Simulation*>(glfwGetWindowUserPointer(window));
	Camera* pCam = pSim->pCamera;
	Object* pStars = pSim->pStarsObj;

	vmath::vec3& cam_velocity = pCam->GetVelocity();
	float speed = 1.0f, rspeed = 45.f;
	bool bPress = (action == GLFW_PRESS | action == GLFW_REPEAT);

	switch(key)
	{
	case GLFW_KEY_SPACE:
		cam_velocity = bPress ? (speed * vmath::vec3(0.f, 1.f, 0.f)) : (0.f * cam_velocity);
		break;
	case GLFW_KEY_LEFT_CONTROL:
		cam_velocity = bPress ? (-speed * vmath::vec3(0.f, 1.f, 0.f)) : (0.f * cam_velocity);
		break;
	case GLFW_KEY_D:
		*(pCam->GetYawSpeed()) = bPress ? -rspeed : 0.f;
		break;
	case GLFW_KEY_A:
		*(pCam->GetYawSpeed()) = bPress ? rspeed : 0.f;
		break;
	case GLFW_KEY_W:
		cam_velocity = bPress ? (speed * pCam->GetLookVector()) : (0.f * cam_velocity);
		break;
	case GLFW_KEY_S:
		cam_velocity = bPress ? (-speed * pCam->GetLookVector()) : (0.f * cam_velocity);
		break;
	case GLFW_KEY_E:
		*(pCam->GetPitchSpeed()) = bPress ? -rspeed : 0.f;
		break;
	case GLFW_KEY_Q:
		*(pCam->GetPitchSpeed()) = bPress ? rspeed : 0.f;
		break;
	case GLFW_KEY_F:
		pStars->AddVertex(vmath::vec4(0.f, 0.f, 0.f, 1.f),
				  vmath::Tvec4<unsigned char>(255, 255, 0, 255));
		break;
	default: [[likely]]
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
		m_wyhash64 = 0xFeedFaceDeadBeef; //This is the seed
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

	glfwWindowHint(GLFW_SAMPLES, 4);
	glEnable(GL_MULTISAMPLE);

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

RandGen g_randgen;

void Ellipse(float* x, float* y, float* z, float t,
	     float a, float b, float cx, float cy, float cz,
	     const vmath::Tquaternion<float>& rot
	)
{
	vmath::vec4 coord(
		(a * cos(t)),
		0.f,
		(b * sin(t)),
		0.f
		);
	coord = rot * coord * rot.inverse();

	//Perform quaternion rotation while vector is at origin,
	//then translate for offset
	*x = coord[0] + cx;
	*y = coord[1] + cy;
	*z = coord[2] + cz;
}

struct Planetoid
{
	Planetoid(float x, float y, float z,
		  unsigned char cr, unsigned char cg, unsigned char cb,
		  float lasta)
	{
		float theta = ((g_randgen.RandDouble(20) - 10.f) * PI)/180.f;
		vmath::vec3 randuv(
			g_randgen.RandDouble(50),
			g_randgen.RandDouble(50),
			g_randgen.RandDouble(50)
			);
		randuv = vmath::normalize(randuv);

		float stf = sin(theta/2.f);
		rot = vmath::Tquaternion<float>(
			randuv[0] * stf,
			randuv[1] * stf,
			randuv[2] * stf,
			1.f * cos(theta/2.f));
		invrot = rot.inverse();

		bool bNeg = (g_randgen.PRNG64() & 255) > 128;
		//anglerate = (PI/180.f) * (g_randgen.RandDouble(30) + 5.f) * (bNeg ? -1.f : 1.f);


		float first = g_randgen.RandDouble(5)/100.f + 0.02f + lasta;
		float second = first + g_randgen.RandDouble(3)/100.f;
		float avgr = (first + second) / 2.f;
		//anglerate = (1.f / (avgr * sqrt(avgr))) * (bNeg ? -1.f : 1.f);
		anglerate = 1/sqrt(avgr); // by Kepler's 2nd law of motion; orbital speed decreases with distance from star
		bool bFlip = (g_randgen.PRNG64() & 255) > 128;
		a = bFlip ? first : second;
		b = bFlip ? second : first;

		c = g_randgen.RandDouble(10)/100.f + 0.05f + lasta;

		//phi = g_randgen.RandDouble(30) * (PI/180.f);

		t = (float) g_randgen.RandDouble(157)/100.f;
		Ellipse(&sx, &sy, &sz, t, a, b, x, y, z, rot);

		red = cr;
		green = cg;
		blue = cb;
		alpha = 255;
	}
	void CalculatePosition(float t, vmath::vec3& offset)
	{
		//This takes t
		Ellipse(&offset[0], &offset[1], &offset[2],
			t, a, b, sx, sy, sz, rot);

	}
	void CalculateOffset(float t_del, vmath::vec3& offset)
	{
		float dtheta = t_del * anglerate;
		//Use derivative of ellipse equation
		vmath::Tquaternion<float> off(
			a * -sin(t) * dtheta,
			0.f * dtheta,
			b * cos(t) * dtheta,
			1.f);

		vmath::vec4 result = rot * off * invrot;
		offset[0] = result[0];
		offset[1] = result[1];
		offset[2] = result[2];
		t += dtheta;
	}

	float sx, sy, sz; //Center
	float a, b, c, phi, t, anglerate;
	unsigned char red, green, blue, alpha;
	vmath::Tquaternion<float> rot, invrot;
};

void CreateEllipse(Object& obj, float ox, float oy, float oz,
		   float a, float b, float c,
		   float phi, vmath::Tquaternion<float>& rot)
{
	int segments = 32;
	constexpr float TWOPI = PI * 2.f;
	unsigned char cval = 64;
	for(int idx = 1; idx <= segments; ++idx)
	{
		float t = (TWOPI/segments) * idx;
		float x = 0.f, y = 0.f, z = 0.f;
		Ellipse(&x, &y, &z, t, a, b, ox, oy, oz, rot);
		obj.AddVertex(vmath::vec4(x, y, z, 1.f),
			      vmath::Tvec4<unsigned char>(cval, cval, cval, 255)
			);

		t = (TWOPI/segments) * (idx - 1);
		Ellipse(&x, &y, &z, t, a, b, ox, oy, oz, rot);

		obj.AddVertex(vmath::vec4(x, y, z, 1.f),
			      vmath::Tvec4<unsigned char>(cval, cval, cval, 255)
			);
	}
}

int main()
{

	GLFWwindow* window = 0;
	if(InitGL(&window) < 0)
	{
		fprintf(stderr, "Initialization failed.\n");
		return 0;
	}

	struct Vertex axes[6] = {
		{{255, 0, 0, 255}, {-1.f, 0.f, 0.f, 1.f}},
		{{255, 0, 0, 255}, {1.f, 0.f, 0.f, 1.f}},

		{{0, 255, 0, 255}, {0.f, 1.f, 0.f, 1.f}},
		{{0, 255, 0, 255}, {0.f, -1.f, 0.f, 1.f}},

		{{0, 0, 255, 255}, {0.f, 0.f, -1.f, 1.f}},
		{{0, 0, 255, 255}, {0.f, 0.f, 1.f, 1.f}}
	};

	Camera camera;
	camera.SetPosition(vmath::vec3(0.f, 1.f, -2.f));
	camera.LookAt(vmath::vec3(0.f, 0.f, 0.f));

	Object axesobj(GL_LINES), stars_obj(GL_POINTS), edges_obj(GL_LINES);
	Object planetoidobj(GL_POINTS);
	Object orbitsobj(GL_LINES);


	struct Simulation sim;
	sim.pCamera = &camera;
	sim.pStarsObj = &stars_obj;
	glfwSetWindowUserPointer(window, &sim);

	vmath::mat4 scale = vmath::scale(1.f, 1.f, 1.f);
	edges_obj.SetObjectTransform(scale);
	stars_obj.SetObjectTransform(scale);
	axesobj.SetObjectTransform(scale);
	planetoidobj.SetObjectTransform(scale);
	orbitsobj.SetObjectTransform(scale);

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
	vmath::Tquaternion<float> pitchrot =
		vmath::Tquaternion<float>(1.f * sin(theta/2.f), 0.f, 0.f, 1.f * cos(theta/2.f));
	vmath::Tquaternion<float> invpitchrot = pitchrot.inverse();

	printf("%f %f %f %f\n", rotation[0], rotation[1], rotation[2], rotation[3]);
	printf("%f %f %f %f\n", inverse[0], inverse[1], inverse[2], inverse[3]);
	std::vector<Planetoid> planetoids;


	std::vector<vmath::vec3> others;
	float mindist = 0.6f; //minimum distance between stars
	for(int idx = 0; idx < 12; ++idx)
	{
		float x = g_randgen.RandDouble(200)/100.f - 1.f;
		float y = g_randgen.RandDouble(200)/100.f - 1.f;
		float z = g_randgen.RandDouble(200)/100.f - 1.f;
		vmath::vec3 testpoint(x, y, z);
		float closest = -1.f;
		for(int i = 0; i < others.size(); ++i)
		{
			float dist = vmath::distance(others[i], testpoint);
			if(closest < 0.f || dist < closest)
			{
				closest = dist;
			}
		}
		if(closest >= 0.f && closest <= mindist)
		{
			printf("%f < %f\n", closest, mindist);
			continue;
		}

		unsigned char red = (g_randgen.PRNG64() + 128) & 255;
		unsigned char green = (g_randgen.PRNG64() + 128) & 255;
		unsigned char blue = (g_randgen.PRNG64() + 128) & 255;

		others.emplace_back(vmath::vec3(x, y, z));
		stars_obj.AddVertex(vmath::vec4(x, y, z, 1.f),
				    vmath::Tvec4<unsigned char>(red, green, blue, 255));
		int numplanets = (int) g_randgen.RandDouble(7) + 1;
		float lasta = 0.05f;
		for(int i = 0; i < numplanets; ++i)
		{
			Planetoid planet(x, y, z, red, green, blue, lasta);
			planetoids.emplace_back(planet);
			planetoidobj.AddVertex(vmath::vec4(planet.sx, planet.sy, planet.sz, 1.f),
					       vmath::Tvec4<unsigned char>(planet.red,
									   planet.green,
									   planet.blue,
									   planet.alpha)
				);
			CreateEllipse(orbitsobj, x, y, z,
				      planet.a, planet.b, planet.c,
				      planet.phi, planet.rot);
			lasta = planet.a;
		}
	}

		orbitsobj.InitBuffer();
	orbitsobj.LoadShaders("orbit.vert", "axes.frag");
	planetoidobj.InitBuffer();
	planetoidobj.LoadShaders("planetoid.vert", "stars.frag");

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

	std::vector<Object*> scene_objs;
	scene_objs.push_back(&axesobj);
	scene_objs.push_back(&stars_obj);
	scene_objs.push_back(&orbitsobj);
	scene_objs.push_back(&edges_obj);


	std::vector<Vertex>& verts = planetoidobj.GetVerts();
	while(!glfwWindowShouldClose(window))
	{
		clock_gettime(CLOCK_MONOTONIC, &t_a);
		glViewport( 0, 0, 1024, 1024);

		// wipe the drawing surface clear
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		for(Object* pObj : scene_objs)
		{
			pObj->Draw(&camera);
			//pObj->Rotate(rotation, inverse);
			pObj->UpdateBuffer();
		}

		planetoidobj.Draw(&camera);
		//planetoidobj.Rotate(rotation, inverse);

		glfwSwapBuffers(window);
		glfwPollEvents();
		//glfwWaitEvents();
		clock_gettime(CLOCK_MONOTONIC, &t_b);
		t_del = TimeDiffSecs(&t_b, &t_a);


		std::vector<Vertex>& planetverts = planetoidobj.GetVerts();
		vmath::vec3 offset;
		for(int i = 0; i < planetoids.size(); ++i)
		{
			planetoids[i].CalculateOffset(t_del, offset);
			planetverts[i].vertex[0] += offset[0];
			planetverts[i].vertex[1] += offset[1];
			planetverts[i].vertex[2] += offset[2];
		}
		planetoidobj.UpdateBuffer();


		//HACK: Don't compare a float for equality
		if(*camera.GetYawSpeed() != 0.f) [[unlikely]]
		{
			float dr = (t_del * (*camera.GetYawSpeed()))/2.f;

			vmath::vec3 localy(vmath::cross(camera.GetLookVector(),
							camera.GetLeftVector()
						   )
				);

			localy = vmath::normalize(localy);
			rotation[0] = localy[0] * sin(dr * PI/180.f);
			rotation[1] = localy[1] * sin(dr * PI/180.f); //Adjust rotation amount for time delta
			rotation[2] = localy[2] * sin(dr * PI/180.f);
			rotation[3] = cos(dr * PI/180.f);

			inverse = rotation.inverse();
			camera.Rotate(rotation, inverse);
		}

		if(*camera.GetPitchSpeed() != 0.f) [[unlikely]]
		{
			float dr = (t_del * (*camera.GetPitchSpeed()))/2.f;
			vmath::vec3 localx(camera.GetLeftVector());

			localx = vmath::normalize(localx);
			pitchrot[0] = localx[0] * sin(dr * PI/180.f);
			pitchrot[1] = localx[1] * sin(dr * PI/180.f);
			pitchrot[2] = localx[2] * sin(dr * PI/180.f);
			pitchrot[3] = cos(dr * PI/180.f);
			invpitchrot = pitchrot.inverse();
			camera.Rotate(pitchrot, invpitchrot);
		}
		camera.Move(camera.GetVelocity() * t_del);
		camera.LookAtTarget();
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
