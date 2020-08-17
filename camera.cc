#include "camera.h"

Camera::Camera(float fov, float aspectratio,
	       float zfar, float znear)
{
	m_velocity = vmath::vec3(0.f, 0.f, 0.f);
	m_radians_ps = 0.f;
	m_view = vmath::mat4::identity();
	m_proj = vmath::mat4::identity();
	m_proj = vmath::perspective(fov, aspectratio, znear, zfar)  * m_proj;
	SetUp(vmath::vec3(0.f, 1.f, 0.f)); //positive y axis is up by default
	SetPosition(vmath::vec3(0.f, 0.f, 0.f));
}
