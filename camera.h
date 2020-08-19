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
		m_target += offset;
	}

       	void Rotate(const vmath::Tquaternion<float>& rotation,
		    const vmath::Tquaternion<float>& inverse);

	void SetPosition(const vmath::vec3& pos)
	{
		m_target -= m_eye;
		m_eye = pos;
		m_target += m_eye;
	}

	void SetUp(const vmath::vec3& up)
	{
		m_up = up;
	}

	vmath::vec3 GetUpVector()
	{
		return m_up;
	}

	void LookAt(const vmath::vec3& target);
	void LookAtTarget();

	const vmath::mat4& GetViewTransform() const
	{
		return m_view;
	}

	const vmath::mat4& GetProjectionTransform() const
	{
		return m_proj;
	}

	vmath::vec3& GetVelocity()
	{
		return m_velocity;
	}

	const vmath::vec3 GetLookVector()
	{
		//return m_forward;
		return vmath::normalize(m_target - m_eye);
	}

	const vmath::vec3& GetLeftVector()
	{
		return m_left;
	}

	//Separate yaw and pitch speeds are kinda dumb
	float* GetYawSpeed()
	{
		return &m_yawspeed;
	}

	float* GetPitchSpeed()
	{
		return &m_pitchspeed;
	}
private:
	vmath::vec3 m_up, m_left, m_forward, m_target, m_eye, m_velocity;
	vmath::mat4 m_view, m_proj;
	float m_zfar, m_znear;
	float m_yawspeed, m_pitchspeed;
};


#endif
