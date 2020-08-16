// MatTest.cpp : Defines the entry point for the console application.
//

#include <xmmintrin.h>
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <smmintrin.h>
union m128union
{
	__m128 vec;
	float array[4];
};

void VecMatrixMultiply(const float* vec, float (*matrix)[4], float* result)
{
	for (unsigned char i = 0; i < 4; ++i)
	{
		const float vec_el = vec[i];
		const float* const matrix_row = matrix[i];

		result[0] += vec_el * matrix_row[0];
		result[1] += vec_el * matrix_row[1];
		result[2] += vec_el * matrix_row[2];
		result[3] += vec_el * matrix_row[3];
	}
}

void SSEVecMatrixMultiply(const float* vec, float(*matrix)[4], float* result)
{
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

	//Simultaneously multiply each row of the matrix by each of the vector's elements
	const __m128 irow0 = _mm_mul_ps(mvec0, row0);
	const __m128 irow1 = _mm_mul_ps(mvec1, row1);
	const __m128 irow2 = _mm_mul_ps(mvec2, row2);
	const __m128 irow3 = _mm_mul_ps(mvec3, row3);

	//Add the multiplied rows up
	__m128 ires = _mm_add_ps(irow0, irow1);
	ires = _mm_add_ps(ires, irow2);
	ires = _mm_add_ps(ires, irow3);

	//Load the result back into RAM
	_mm_store_ps(result, ires);
}

void AVXVecMatrixMultiply(const float* vec, float(*matrix)[4], float* result)
{
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

void AVX2VecMatrixMultiply(const float* vec, float(*matrix)[4], float* result,
	const float* vec2, float (*matrix2)[4], float* result2)
{
	//Load the SIMD registers from RAM
	const __m256 mvec = _mm256_loadu2_m128(vec, vec2);

	const __m256 row0 = _mm256_loadu2_m128(matrix[0], matrix2[0]);
	const __m256 row1 = _mm256_loadu2_m128(matrix[1], matrix2[1]);
	const __m256 row2 = _mm256_loadu2_m128(matrix[2], matrix2[2]);
	const __m256 row3 = _mm256_loadu2_m128(matrix[3], matrix2[3]);

	const __m256 mvec0 = _mm256_shuffle_ps(mvec, mvec, _MM_SHUFFLE(0, 0, 0, 0));
	const __m256 mvec1 = _mm256_shuffle_ps(mvec, mvec, _MM_SHUFFLE(1, 1, 1, 1));
	const __m256 mvec2 = _mm256_shuffle_ps(mvec, mvec, _MM_SHUFFLE(2, 2, 2, 2));
	const __m256 mvec3 = _mm256_shuffle_ps(mvec, mvec, _MM_SHUFFLE(3, 3, 3, 3));

	__m256 ires = _mm256_set1_ps(0.f);
	ires = _mm256_fmadd_ps(mvec0, row0, ires);
	ires = _mm256_fmadd_ps(mvec1, row1, ires);
	ires = _mm256_fmadd_ps(mvec2, row2, ires);
	ires = _mm256_fmadd_ps(mvec3, row3, ires);
	_mm256_storeu2_m128(result, result2, ires);
}

void AlterVec(float vec[4], float val = 1.f)
{
	const __m128 mvec = _mm_load_ps(vec);
	const __m128 madd = _mm_set1_ps(val);
	const __m128 mres = _mm_add_ps(mvec, madd);

	_mm_store_ps(vec, mres);
}

void AlterVec(const float vec[4], float outvec[4], float val = 1.f)
{
	const __m128 mvec = _mm_load_ps(vec);
	const __m128 madd = _mm_set1_ps(val);
	const __m128 mres = _mm_add_ps(mvec, madd);

	_mm_store_ps(outvec, mres);
}

void AlterMatrix(float(*mat)[4], float val = 1.f)
{
	const __m256 row01 = _mm256_loadu_ps(mat[0]);
	const __m256 row23 = _mm256_loadu_ps(mat[2]);
	const __m256 addition = _mm256_set1_ps(val);

	const __m256 upper = _mm256_add_ps(row01, addition);
	const __m256 lower = _mm256_add_ps(row23, addition);

	_mm256_store_ps(mat[0], upper);
	_mm256_store_ps(mat[2], lower);
}

void AlterMatrix(const float (*mat)[4], float (*matout)[4], float val = 1.f)
{
	const __m256 row01 = _mm256_loadu_ps(mat[0]);
	const __m256 row23 = _mm256_loadu_ps(mat[2]);
	const __m256 addition = _mm256_set1_ps(val);

	const __m256 upper = _mm256_add_ps(row01, addition);
	const __m256 lower = _mm256_add_ps(row23, addition);

	_mm256_store_ps(matout[0], upper);
	_mm256_store_ps(matout[2], lower);
}

void AddVector(float* a, const float* b)
{
	const __m128 mvec_a = _mm_load_ps(a);
	const __m128 mvec_b = _mm_load_ps(b);

	const __m128 mres = _mm_add_ps(mvec_a, mvec_b);
	_mm_store_ps(a, mres);
}

void TestFPU(unsigned int iterations, float* finalresult)
{
	float vec[4] = { 1.f, 2.f, 3.f, 4.f };
	float mat[4][4] = {
		{ 1.f, 5.f, 9.f, 13.f },
		{ 2.f, 6.f, 10.f, 14.f },
		{ 3.f, 7.f, 11.f, 15.f },
		{ 4.f, 8.f, 12.f, 16.f },
	};

	for (unsigned int i = 0; i < iterations; ++i)
	{
		float result[4] = { 0.f };
		VecMatrixMultiply(vec, mat, result);
		AddVector(finalresult, result);
		AlterMatrix(mat);
		AlterVec(vec);
	}

}

void TestSSE(unsigned int iterations, float* finalresult)
{
	float vec[4] = { 1.f, 2.f, 3.f, 4.f };
	float mat[4][4] = {
		{ 1.f, 5.f, 9.f, 13.f },
		{ 2.f, 6.f, 10.f, 14.f },
		{ 3.f, 7.f, 11.f, 15.f },
		{ 4.f, 8.f, 12.f, 16.f },
	};

	for (unsigned int i = 0; i < iterations; ++i)
	{
		float result[4] = { 0.f };
		SSEVecMatrixMultiply(vec, mat, result);
		AddVector(finalresult, result);
		AlterMatrix(mat);
		AlterVec(vec);
	}
}

void TestAVX(unsigned int iterations, float* finalresult)
{
	float vec[4] = { 1.f, 2.f, 3.f, 4.f };
	float mat[4][4] = {
		{ 1.f, 5.f, 9.f, 13.f },
	{ 2.f, 6.f, 10.f, 14.f },
	{ 3.f, 7.f, 11.f, 15.f },
	{ 4.f, 8.f, 12.f, 16.f },
	};

	for (unsigned int i = 0; i < iterations; ++i)
	{
		float result[4] = { 0.f };
		AVXVecMatrixMultiply(vec, mat, result);
		AddVector(finalresult, result);
		AlterMatrix(mat);
		AlterVec(vec);
	}
}

void TestAVX2(unsigned int iterations, float* finalresult)
{
	float vec[4] = { 1.f, 2.f, 3.f, 4.f };
	float mat[4][4] = {
		{ 1.f, 5.f, 9.f, 13.f },
	{ 2.f, 6.f, 10.f, 14.f },
	{ 3.f, 7.f, 11.f, 15.f },
	{ 4.f, 8.f, 12.f, 16.f },
	};

	for (unsigned int i = 0, eveniterations = iterations & -2 ; i < eveniterations; i += 2)
	{
		float mat2[4][4] = { 0.f };
		float result[4] = { 0.f }, result2[4] = { 0.f };
		float vec2[4] = { 0.f };
		AlterVec(vec, vec2);
		AlterMatrix(mat, mat2);

		AVX2VecMatrixMultiply(vec, mat, result,
			vec2, mat2, result2);

		AddVector(finalresult, result);
		AddVector(finalresult, result2);

		AlterMatrix(mat, 2.f);
		AlterVec(vec, 2.f);
	}

	if (iterations & 1)
	{
		float result[4] = { 0.f };
		AVXVecMatrixMultiply(vec, mat, result);
		AddVector(finalresult, result);
	}
}

float DotProduct(float* a, float* b)
{
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
}

void Printm128(const __m128& mvec)
{
	float temp[4];
	_mm_store_ps(temp, mvec);
	printf("%f %f %f %f\n", temp[0], temp[1], temp[2], temp[3]);
}

float SSEDotProduct(float* a, float* b)
{
	const __m128 mvec_a = _mm_load_ps(a);
	const __m128 mvec_b = _mm_load_ps(b);
	union m128union mresult;
	mresult.vec = _mm_dp_ps(mvec_a, mvec_b, 0x77);
	return mresult.array[0];
}

void TestDotProduct(const unsigned int iterations, float* sum)
{
	*sum = 0.f;
	float a[4] = { 1.f, 2.f, 3.f, 0.f };
	float b[4] = { 1.f, 2.f, 3.f, 0.f };

	for (unsigned int i = 0; i < iterations; ++i)
	{
		*sum += DotProduct(a, b);
		AlterVec(a);
		AlterVec(b);
	}
}

void TestSSEDotProductB(const unsigned int iterations, float* sum)
{
	*sum = 0.f;
	float a[4] = { 1.f, 2.f, 3.f, 0.f };
	float b[4] = { 1.f, 2.f, 3.f, 0.f };

	for (unsigned int i = 0; i < iterations; ++i)
	{
		*sum += SSEDotProduct(a, b);
		AlterVec(a);
		AlterVec(b);
	}
}

void TestSSEDotProduct(const unsigned int iterations, float* sum)
{
	__m128 mvec_a = _mm_set_ps(1.f, 2.f, 3.f, 0.f);
	__m128 mvec_b = _mm_set_ps(1.f, 2.f, 3.f, 0.f);
	const __m128 minc = _mm_set_ps(1.f, 1.f, 1.f, 1.f);

	union m128union msum;
	msum.vec = _mm_set1_ps(0.f);

	for (unsigned int i = 0; i < iterations; ++i)
	{
		const __m128 mresult = _mm_dp_ps(mvec_a, mvec_b, 0xff);
		msum.vec = _mm_add_ps(msum.vec, mresult);

		mvec_a = _mm_add_ps(mvec_a, minc);
		mvec_b = _mm_add_ps(mvec_b, minc);
	}
	*sum = msum.array[0];
}

int main()
{
	float sum1 = 0.f, sum2 = 0.f;
	unsigned int iterations = 1000;
	TestDotProduct(iterations, &sum1);
	TestSSEDotProductB(iterations, &sum2);
	printf("%f %f\n", sum1, sum2);


	float finalresultA[4] = { 0.f }, finalresultB[4] = { 0.f }, finalresultC[4] = { 0.f }, finalresultD[4] = { 0.f };

	TestFPU(iterations, finalresultA);
	TestSSE(iterations, finalresultB);
	TestAVX(iterations, finalresultC);
	TestAVX2(iterations, finalresultD);

	printf("%f %f %f %f\n", finalresultA[0], finalresultA[1], finalresultA[2], finalresultA[3]);
	printf("%f %f %f %f\n", finalresultB[0], finalresultB[1], finalresultB[2], finalresultB[3]);
	printf("%f %f %f %f\n", finalresultC[0], finalresultC[1], finalresultC[2], finalresultC[3]);
	printf("%f %f %f %f\n", finalresultD[0], finalresultD[1], finalresultD[2], finalresultD[3]);

    return 0;
}
