#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define SIZE 1024
#define DELTA_X 1.0
#define DIFF 0.1
#define ITERATIONS 100
#define DELTA_T 0.1

#define UI 0

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif


// unrolled matrix into one vector
int cellID(int x, int y)
{
	return SIZE*x + y;
}
// /*
#ifdef GPU
// update updates matrix one iteration. uses GPU
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
void update(double M[SIZE*SIZE], double cache[SIZE*SIZE], int threads, int iterations, int mode)
{
	#ifdef GPU
	//cuda mode
	if(mode == 1)
	{
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
			int blk_count = (SIZE*SIZE + (threads-1))/threads;
			kernel_update<<<blk_count,threads>>>(a, b);
			cudaDeviceSynchronize();
			kernel_update<<<blk_count,threads>>>(b, a);
			cudaDeviceSynchronize();
		}

		// pull data from GPU
		cudaMemcpy(M, a, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost);

		cudaFree(a); cudaFree(b);

		// done.
		return;
	}
	#endif


	if(mode == 0)
	{
	// swap pointers instead of matrcies
	double* t0 = M;
	double* t1 = cache;

	if(iterations % 2) iterations--;	//rounded down to even
	for(int t = 0; t < iterations; t++)
	{
		double difusion_sum = 0.0;

		//for every cell
		#pragma omp parallel for reduction(+:difusion_sum) num_threads(threads)
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
	}

	if(mode == 2)
	{
	// lines processed per process
	int lines_per_thread = SIZE/threads;
	int l_per_t = lines_per_thread;
	// lines missing
	int over = SIZE - (lines_per_thread*threads);
	int ovr = over;

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// printf("process: %d\n", rank);

	// take one extra line, if there are enough
	if(rank < over)
	{
		l_per_t++;
		ovr = 0;
	}
	int low_index  = ((rank + 0)*l_per_t + ovr)*SIZE;
	int high_index = ((rank + 1)*l_per_t + ovr)*SIZE;


	// swap pointers instead of matrcies
	double* t0 = M;
	double* t1 = cache;

	MPI_Status stat;
	int err;

	if(iterations % 2) iterations--;	//rounded down to even
	for(int t = 0; t < iterations; t++)
	{
		double difusion_sum = 0.0;


		//for every cell
		for(int index = low_index; index < high_index; index++)
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


		// synchronize frontiers
		if(rank % 2) // if odd
		{
			//send low line
			// if(rank > 0) // it can't be 0
			{
				err = MPI_Sendrecv (
				&(t1[low_index     ]), SIZE, MPI_DOUBLE, rank-1, t,
				&(t1[low_index-SIZE]), SIZE, MPI_DOUBLE, rank-1, t,
				MPI_COMM_WORLD, &stat);
			}
			// send high line
			if(rank < threads-1)
			{
				err = MPI_Sendrecv (
				&(t1[high_index-SIZE]), SIZE, MPI_DOUBLE, rank+1, t,
				&(t1[high_index     ]), SIZE, MPI_DOUBLE, rank+1, t,
				MPI_COMM_WORLD, &stat);
			}
		}
		else
		{
			// send high line
			if(rank < threads-1)
			{
				err = MPI_Sendrecv (
				&(t1[high_index-SIZE]), SIZE, MPI_DOUBLE, rank+1, t,
				&(t1[high_index     ]), SIZE, MPI_DOUBLE, rank+1, t,
				MPI_COMM_WORLD, &stat);
			}
			//send low line
			if(rank > 0)
			{
				err = MPI_Sendrecv (
				&(t1[low_index     ]), SIZE, MPI_DOUBLE, rank-1, t,
				&(t1[low_index-SIZE]), SIZE, MPI_DOUBLE, rank-1, t,
				MPI_COMM_WORLD, &stat);
			}
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

	// sync up processes, send updated lines
	for(int ranks = 0; ranks < threads; ranks++)
	{
		if(ranks < over) {l_per_t = lines_per_thread + 1;	ovr =    0;}
		else             {l_per_t = lines_per_thread + 0;	ovr = over;}
		// printf("%d[%d]\n", ((l_per_t*ranks) + ovr)*SIZE,  l_per_t*SIZE);
		err = MPI_Bcast(&M[((l_per_t*ranks) + ovr)*SIZE], l_per_t*SIZE, MPI_DOUBLE, ranks, MPI_COMM_WORLD);
	}
	// printf("\n");
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

void run_mode
(
	int mode, int threads, int iterations, 
	double* M0, double* cache,
	FILE* file,
	double* SCM
);
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

	int size_of_cluster = 1;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size_of_cluster);
	// printf("processes: %d\n", size_of_cluster);

	// serial check, disable threads and run normally
	for(int i = 0; i < SIZE*SIZE; i++)	SCM[i] = 0.0;
	SCM[cellID(SIZE/2, SIZE/2)] = 1.0;
	update(SCM, cache, 1, iterations, 0);
	printf("check matrix created\n");


	FILE* file = fopen("result", "w");
	// for each mode, run tests
	if(size_of_cluster == 1)	// this is  a normal CUDA/OMP run
	{
		#ifdef GPU
		for(int threads = 1; threads < 65; threads++)
		{
			run_mode(1, threads, iterations, M0, cache, file, SCM);
		}
		#endif
		// for each number of threads, run test
		for(int threads = 1; threads < 17; threads++)
		{
			run_mode(0, threads, iterations, M0, cache, file, SCM);
		}
	}
	else
	{
		//this is a MPI test
		run_mode(2, size_of_cluster, iterations, M0, cache, file, SCM);

	}

	MPI_Finalize();
	fclose(file);


	return 0;
}


void run_mode(int mode, int threads, int iterations, double* M0, double* cache, FILE* file, double* SCM)
{
	// printf("cooldown... [press enter to continue]");
	// getc(stdin);
	// printf("start.\n");
	fprintf(file, "%d; %d; ", mode, threads);

	double times[12];
	for(int reps = 0; reps < 12; reps++)
	{
		for(int i = 0; i < SIZE*SIZE; i++)	M0[i] = 0.0;
		for(int i = 0; i < SIZE*SIZE; i++)	cache[i] = 0.0;
		M0[cellID(SIZE/2, SIZE/2)] = 1.0;


		struct timespec start, end;

		#ifndef GPU
		clock_gettime(CLOCK_MONOTONIC, &start);
		#else
		cudaDeviceSynchronize();
		timespec_get(&start, TIME_UTC);
		#endif

		update(M0, cache, threads, iterations, mode);

		#ifndef GPU
		clock_gettime(CLOCK_MONOTONIC, &end);
		#else
		timespec_get(&end, TIME_UTC);
		#endif

		long long time = 1000000000*(end.tv_sec-start.tv_sec) + (end.tv_nsec-start.tv_nsec);
		// fprintf(file, "\nM[%d][%d]: %f", SIZE/2, SIZE/2, M0[cellID(SIZE/2, SIZE/2)]);
		// fprintf(file, "\telapsed seconds: %lld,%09lld", time/1000000000, time%1000000000); 
		times[reps] = time/1000000000.0;
		// fprintf(file, "\telapsed seconds: %lld.%03lld %03lld %03lld", 
		// (time/1000000000)%1000, 
		// (time/1000000)%1000, 
		// (time/1000)%1000, 
		// (time)%1000
		// );
	}
	//get rid of the biggest and smallest values
	{
		int biggest = 0, smallest = 0;
		for(int i = 1; i < 12; i++)
		{
			if(times[smallest] > times[i]) smallest = i;
			if(times[biggest ] < times[i]) biggest  = i;
		}
		double swap;
		swap = times[10]; times[10] = times[biggest ]; times[biggest ] = swap;
		swap = times[11]; times[11] = times[smallest]; times[smallest] = swap;
	}
	//calculate the medium
	double med = 0;
	for(int i = 0; i < 10; i++) med += times[i];
	med = med /10;
	//and then standard deviation
	double std = 0;
	for(int i = 0; i < 10; i++)
	{
		double t = (times[i] - med);
		std += t*t;
	}
	std = sqrt(std / 10.0);

	fprintf(file, "%.9lf; %.9lf\n", med, std);
	printf("%d done\n", threads);
	// // verify
	// printf("\nM0");
	// for(int i = 0; i < SIZE*SIZE; i++){if((i%SIZE)==0)printf("\n"); printf("%lf ", M0[i]);}
	// printf("\n");
	// printf("\nSCM");
	// for(int i = 0; i < SIZE*SIZE; i++){if((i%SIZE)==0)printf("\n"); printf("%lf ",SCM[i]);}
	// printf("\n");
	if(compare_matrices(M0, SCM)) printf("serial-parralel Match.\n");
	else						  printf("serial-parralel ERROR mismatch!\n");

	return;
}