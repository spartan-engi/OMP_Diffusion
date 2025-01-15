#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 300
#define DELTA_X 1.0
#define DIFF 0.1
#define ITERATIONS 100
#define DELTA_T 0.1

#define UI 0


// unrolled matrix into one vector
int cellID(int x, int y)
{
	return SIZE*x + y;
}
// update updates matrix with all iterations. uses second matrix as working memory
void update(double M[SIZE*SIZE], double cache[SIZE*SIZE], int mode)
{
	// swap pointers instead of matrcies
	double* t0 = M;
	double* t1 = cache;

	for(int t = 0; t < ITERATIONS; t++)
	{
		double difusion_sum = 0.0;

		#pragma omp parallel for reduction(+:difusion_sum) num_threads(mode)
		for(int i = 1; i < SIZE-1; i++)
		{
			for(int j = 1; j < SIZE-1; j++)
			{
				// change
				double diff = 0.0;
				diff -= 4.0*(t0[cellID(i,j)]);
				diff += (t0[cellID(i-1,j  )]);
				diff += (t0[cellID(i  ,j-1)]);
				diff += (t0[cellID(i+1,j  )]);
				diff += (t0[cellID(i  ,j+1)]);
				diff = diff * DELTA_T*DIFF*(1/(DELTA_X*DELTA_X));

				difusion_sum += ((diff > 0) ? diff : -diff);

				t1[cellID(i, j)] = diff + t0[cellID(i, j)];
			}
		}

		#if UI
		if(!(t % 50)) {printf("i:%3d difference: %lf\n", t, difusion_sum);}
		#endif

		//significant speedup from switching values around
		double* t;
		t  = t0;
		t0 = t1;
		t1 = t ;
	}

	#if(ITERATIONS % 2)
	// extra swap to guarantee the right matrix is result
	for(int i = 1; i < SIZE-1; i++)
	{
		for(int j = 1; j < SIZE-1; j++)
		{
			M[cellID(i, j)] = cache[cellID(i, j)];
		}
	}
	#endif
	return;
}
// returns 1 if matricies are identical
int compare_matrices(double* M0, double* M1)
{
	int valid = 1;
	for(int i = 0; i < SIZE; i++)
	{
		for(int j = 0; j < SIZE; j++)
		{
			if(M0[cellID(i,j)] != M1[cellID(i,j)])
			{
				return 0;
			}
		}
	}
	return 1;
}

int main(int argc, char* argv[])
{
	int mode = 0;
	if(argc > 1) mode = atoi(argv[1]);
	// double M0[SIZE * SIZE] = {0};
	// double M1[SIZE * SIZE] = {0};
	double* M0 = malloc(sizeof(double)*SIZE*SIZE);
	double* cache = malloc(sizeof(double)*SIZE*SIZE);
	// serial check matrix
	double* SCM = malloc(sizeof(double)*SIZE*SIZE);


	// serial check, disable threads and run normally
	for(int i = 0; i < SIZE*SIZE; i++)	SCM[i] = 0.0;
	SCM[cellID(SIZE/2, SIZE/2)] = 1.0;
	update(SCM, cache, 1);
	printf("check matrix created\n");


	FILE* file = fopen("result", "w");
	for(int reps = 0; reps < 10; reps++)
	{
		for(int i = 0; i < SIZE*SIZE; i++)	M0[i] = 0.0;
		M0[cellID(SIZE/2, SIZE/2)] = 1.0;


		struct timespec start, end;
		clock_gettime(CLOCK_MONOTONIC, &start);

		update(M0, cache, mode);

		clock_gettime(CLOCK_MONOTONIC, &end);


		fprintf(file, "\nM[%d][%d]: %f", SIZE/2, SIZE/2, M0[cellID(SIZE/2, SIZE/2)]);
		long long time = 1000000000*(end.tv_sec-start.tv_sec) + (end.tv_nsec-start.tv_nsec);
		fprintf(file, "\nelapsed seconds: %lld.%03lld %03lld %03lld", 
		(time/1000000000)%1000, 
		(time/1000000)%1000, 
		(time/1000)%1000, 
		(time)%1000
		);
	}
	fclose(file);

	// verify
	if(compare_matrices(M0, SCM)) printf("serial-paralel match.\n");
	else						  printf("serial-paralel mismatch!\n");

	return 0;
}