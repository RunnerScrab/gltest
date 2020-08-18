#include "camera.h"
#include "ssemath.h"

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

void Camera::Rotate(const vmath::Tquaternion<float>& rotation,
		    const vmath::Tquaternion<float>& inverse)
{
	float pos[4] = {m_eye[0], m_eye[1], m_eye[2], 0.f};
	float result[4];
	sse_quat_mul(&rotation[0], pos, result);
	sse_quat_mul(result, &inverse[0], pos);
	m_eye = vmath::vec3(pos[0], pos[1], pos[2]);
	//vmath::vec4 neweye = rotation * vmath::vec4(m_eye[0], m_eye[1], m_eye[2], 0.f) * inverse;
	//m_eye = vmath::vec3(neweye[0], neweye[1], neweye[2]);
}
