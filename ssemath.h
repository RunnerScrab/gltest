#ifndef SSEMATH_H_
#define SSEMATH_H_
#include <immintrin.h>
#include <smmintrin.h>
#define qx(q) q[0]
#define qy(q) q[1]
#define qz(q) q[2]
#define qw(q) q[3]

void PrintQuat(float* quat, const char* str);

inline __m128 SSECrossProduct(__m128 vec_a, __m128 vec_b)
{
	__m128 sh_a = _mm_shuffle_ps(vec_a, vec_a, _MM_SHUFFLE(3, 0, 2, 1));
	__m128 sh_b = _mm_shuffle_ps(vec_b, vec_b, _MM_SHUFFLE(3, 0, 2, 1));
	__m128 result = _mm_sub_ps(_mm_mul_ps(vec_a, sh_b), _mm_mul_ps(vec_b, sh_a));
	return _mm_shuffle_ps(result, result, _MM_SHUFFLE(3, 0, 2, 1));
}

inline void sse_quat_applyrot(const float* a, const float* b, float* out)
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

inline void sse_quat_mul(const float* a, const float* b, float* out)
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
#endif
