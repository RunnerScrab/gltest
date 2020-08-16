#include <cstdio>
#include <ctime>
#include <cmath>

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

int main(void)
{
	RandGen randgen;
	float x = randgen.RandDouble(200)/100.f - 1.f;
	float y = randgen.RandDouble(200)/100.f - 1.f;
	float z = randgen.RandDouble(200)/100.f - 1.f;
	printf("%f, %f, %f\n", x, y, z);
	return 0;
}
