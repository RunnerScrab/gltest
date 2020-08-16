#ifndef CAMERA_H_
#define CAMERA_H_
#include "vmath.h"

class Camera
{
public:
	Camera(float fov = 67.f, float aspectratio = 1.f,
		float zfar = 200.f, float znear = 0.1f);

	void Move(const vmath::vec3& offset)
	{
		m_eye += offset;
	}

	void SetPosition(const vmath::vec3& pos)
	{
		m_eye = pos;
	}

	void SetUp(const vmath::vec3& up)
	{
		m_up = up;
	}

	void LookAt(const vmath::vec3& target)
	{
		m_target = target;
		m_view = vmath::lookat(m_eye, m_target, m_up);
	}

	const vmath::mat4& GetViewTransform() const
	{
		return m_view;
	}

	const vmath::mat4& GetProjectionTransform() const
	{
		return m_proj;
	}
private:
	vmath::vec3 m_up, m_target, m_eye;
	vmath::mat4 m_view, m_proj;
	float m_zfar, m_znear;
};


#endif
