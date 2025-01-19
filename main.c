#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1024
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
// /*
#ifdef GPU
// update updates matrix with all iterations. uses GPU
__global__ void kernel_update(double* in, double* out)
{
	//index that the thread will act upon
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// if and only if it is inside the matrix
	if(index >= SIZE*SIZE) return;

	// change in concentration
	double diff = 0.0;
	// total concentration on the cells up/left/down/right of the middle cell
	if(0 <= (index - SIZE)            ) {diff += in[index - SIZE];}
	if(0 <= (index -    1)            ) {diff += in[index -    1];}
	if(     (index + SIZE) < SIZE*SIZE) {diff += in[index + SIZE];}
	if(     (index +    1) < SIZE*SIZE) {diff += in[index +    1];}
	// minus the concentration of the middle cell
	diff -= 4.0*(in[index]);
	// multiplied by the diffusion coeficient
	diff = diff * DELTA_T*DIFF*(1/(DELTA_X*DELTA_X));

	//becomes the change in concentration of the middle cell
	out[index] = diff + in[index];

	return;
}
#endif
// */
// update updates matrix with all iterations. uses second matrix as working memory
void update(double M[SIZE*SIZE], double cache[SIZE*SIZE], int mode, int iterations)
{
	#ifdef GPU
	//cuda mode
	if(mode < 0)
	{
		mode = -mode;
		// setup buffers to send matrix
		double* a;
		double* b;
		cudaMalloc((void**)&a, sizeof(double)*SIZE*SIZE);
		cudaMalloc((void**)&b, sizeof(double)*SIZE*SIZE);

		// push data to GPU
		cudaMemcpy(a, M, sizeof(double)*SIZE*SIZE, cudaMemcpyHostToDevice);
		// cudaMemcpy(b) //this is a cache, no setup

		// run simulation
		for(int t = 0; t < iterations/2; t++)
		{
			int blk_count = (SIZE*SIZE + (mode-1))/mode;
			kernel_update<<<blk_count,mode>>>(a, b);
			cudaDeviceSynchronize();
			kernel_update<<<blk_count,mode>>>(b, a);
			cudaDeviceSynchronize();
		}

		// pull data from GPU
		cudaMemcpy(M, a, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost);

		cudaFree(a); cudaFree(b);

		// done.
		return;
	}
	#endif


	// swap pointers instead of matrcies
	double* t0 = M;
	double* t1 = cache;

	if(iterations % 2) iterations--;	//rounded down to even
	for(int t = 0; t < iterations; t++)
	{
		double difusion_sum = 0.0;

		//for every cell
		#pragma omp parallel for reduction(+:difusion_sum) num_threads(mode)
		for(int index = 0; index < SIZE*SIZE; index++)
		{
			// change in concentration
			double diff = 0.0;

			// total concentration on the cells up/left/down/right of the middle cell
			if(0 <= (index - SIZE)            ) {diff += (t0[index - SIZE]);}
			if(0 <= (index -    1)            ) {diff += (t0[index -    1]);}
			if(     (index + SIZE) < SIZE*SIZE) {diff += (t0[index + SIZE]);}
			if(     (index +    1) < SIZE*SIZE) {diff += (t0[index +    1]);}
			// minus the concentration of the middle cell
			diff -= 4.0*(t0[index]);
			// multiplied by the diffusion coeficient
			diff = diff * DELTA_T*DIFF*(1/(DELTA_X*DELTA_X));

			difusion_sum += ((diff > 0) ? diff : -diff);

			//becomes the change in concentration of the middle cell
			t1[index] = diff + t0[index];
		}

		#if UI
		if(!(t % 50)) {printf("i:%3d difference: %lf\n", t, difusion_sum);}
		#endif

		//significant speedup from switching values around
		double* swap_pointer;
		swap_pointer  = t0;
		t0 = t1;
		t1 = swap_pointer ;
	}

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
			double p = M0[cellID(i,j)];
			double s = M1[cellID(i,j)];
			double diff = s - p;
			if(diff > 0.000000000000001)
			{
				valid = 0;
			}
		}
	}
	return valid;
}

int main(int argc, char* argv[])
{
	// double M0[SIZE * SIZE] = {0};
	// double M1[SIZE * SIZE] = {0};
	double* M0    = (double*)malloc(sizeof(double)*SIZE*SIZE);
	double* cache = (double*)malloc(sizeof(double)*SIZE*SIZE);
	// serial check matrix
	double* SCM   = (double*)malloc(sizeof(double)*SIZE*SIZE);

	// standard amount of iterations
	// gets rounded down to even
	int iterations = ITERATIONS;

	// serial check, disable threads and run normally
	for(int i = 0; i < SIZE*SIZE; i++)	SCM[i] = 0.0;
	SCM[cellID(SIZE/2, SIZE/2)] = 1.0;
	update(SCM, cache, 1, iterations);
	printf("check matrix created\n");


	FILE* file = fopen("result", "w");
	// for each number, run test
	for(int modes = 1; modes < argc; modes++)
	{
	int mode = atoi(argv[modes]);
	printf("cooldown... [press enter to continue]");
	getc(stdin);
	printf("start:\n");
	fprintf(file, "\nmode:%d", mode);
	for(int reps = 0; reps < 10; reps++)
	{
		for(int i = 0; i < SIZE*SIZE; i++)	M0[i] = 0.0;
		M0[cellID(SIZE/2, SIZE/2)] = 1.0;


		struct timespec start, end;

		#ifndef GPU
		clock_gettime(CLOCK_MONOTONIC, &start);
		#else
		cudaDeviceSynchronize();
		timespec_get(&start, TIME_UTC);
		#endif

		update(M0, cache, mode, iterations);

		#ifndef GPU
		clock_gettime(CLOCK_MONOTONIC, &end);
		#else
		timespec_get(&end, TIME_UTC);
		#endif


		fprintf(file, "\nM[%d][%d]: %f", SIZE/2, SIZE/2, M0[cellID(SIZE/2, SIZE/2)]);
		long long time = 1000000000*(end.tv_sec-start.tv_sec) + (end.tv_nsec-start.tv_nsec);
		fprintf(file, "\telapsed seconds: %lld,%09lld", time/1000000000, time%1000000000); 
		// fprintf(file, "\telapsed seconds: %lld.%03lld %03lld %03lld", 
		// (time/1000000000)%1000, 
		// (time/1000000)%1000, 
		// (time/1000)%1000, 
		// (time)%1000
		// );
	}

	printf("%d done\n", mode);
	// verify
	if(compare_matrices(M0, SCM)) printf("serial-parralel Match.\n");
	else						  printf("serial-parralel ERROR mismatch!\n");
	}
	fclose(file);


	return 0;
}