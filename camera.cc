#include "camera.h"
#include "ssemath.h"
#include <cstring>
#include <cstdio>

Camera::Camera(float fov, float aspectratio,
	       float zfar, float znear)
{
	m_velocity = vmath::vec3(0.f, 0.f, 0.f);
	m_yawspeed = 0.f;
	m_pitchspeed = 0.f;
	m_view = vmath::mat4::identity();
	m_proj = vmath::mat4::identity();
	m_proj = vmath::perspective(fov, aspectratio, znear, zfar)  * m_proj;
	SetUp(vmath::vec3(0.f, 1.f, 0.f)); //positive y axis is up by default
	SetPosition(vmath::vec3(0.f, 0.f, 0.f));
}

void LERP(const vmath::Tquaternion<float>& qa,
	  const vmath::Tquaternion<float>& qb,
	  float beta, vmath::Tquaternion<float>& output)
{
	float cbeta = 1.f - beta;
	vmath::Tquaternion<float> a = (cbeta * qa);
	a += (beta * qb);
	output = a / a.length();
}

void Camera::Rotate(const vmath::Tquaternion<float>& rotation,
		    const vmath::Tquaternion<float>& inverse)
{
	/*
	float pos[4] = {m_target[0] - m_eye[0], m_target[1] - m_eye[1], m_target[2] - m_eye[2], 0.f};
	float result[4];

	sse_quat_mul(&rotation[0], pos, result);
	sse_quat_mul(result, &inverse[0], pos);
	m_target = vmath::vec3(pos[0] + m_eye[0],
			       pos[1] + m_eye[1],
			       pos[2] + m_eye[2]);
	*/
	vmath::vec4 newtarget = rotation * vmath::vec4(
		m_target[0] - m_eye[0],
		m_target[1] - m_eye[1],
		m_target[2] - m_eye[2], 0.f) * inverse;
	//m_eye = vmath::vec3(neweye[0], neweye[1], neweye[2]);
	m_target = vmath::vec3(newtarget[0] + m_eye[0],
			       newtarget[1] + m_eye[1],
			       newtarget[2] + m_eye[2]);

}

void Camera::LookAt(const vmath::vec3& target)
{
	m_target = target;
	m_view = vmath::lookat(m_eye, m_target, m_up);
	m_forward = m_target - m_eye;
	m_left = vmath::normalize(vmath::cross(vmath::vec3(0.f, 1.f, 0.f), m_forward));
	m_up = vmath::normalize(vmath::cross(m_forward, m_left));

/*
	m_left[0] = m_view[0][0];
	m_left[1] = m_view[0][1];
	m_left[2] = m_view[0][2];
	m_up[0] = m_view[1][0];
	m_up[1] = m_view[1][1];
	m_up[2] = m_view[1][2];
	m_forward = vmath::cross(m_up, m_left);
*/
}

void Camera::LookAtTarget()
{
	m_view = vmath::lookat(m_eye, m_target, m_up);
	m_forward = m_target - m_eye;
	m_left = vmath::normalize(vmath::cross(vmath::vec3(0.f, 1.f, 0.f), m_forward));
	m_up = vmath::normalize(vmath::cross(m_forward, m_left));
	/*
	m_left[0] = m_view[0][0];
	m_left[1] = m_view[0][1];
	m_left[2] = m_view[0][2];
	m_up[0] = m_view[1][0];
	m_up[1] = m_view[1][1];
	m_up[2] = m_view[1][2];
	m_forward = vmath::cross(m_up, m_left);
	*/
}
