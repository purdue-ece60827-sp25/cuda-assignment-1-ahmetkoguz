
#include "cudaLib.cuh"
#define DEBUG_PRINT_DISABLE

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size)
		y[i] = scale * x[i] + y[i];
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here

	// #ifndef DEBUG_PRINT_DISABLE 
	// 	printf("vectorsize = %d\n", vectorSize);
	// #endif

	// On host memory
	float *a, *b, *c;

    a = (float*)malloc(vectorSize * sizeof(float));
    b = (float*)malloc(vectorSize * sizeof(float));
    c = (float*)malloc(vectorSize * sizeof(float));

    if (a == NULL || b == NULL || c == NULL) {
        printf("Unable to malloc memory ... Exiting!");
        return -1;
    }

	// Initialize vectors/inputs
	vectorInit(a, vectorSize);
	vectorInit(b, vectorSize);
	float scale = 2.0f;

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" a = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", a[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", b[i]);
		}
		printf(" ... }\n");
	#endif


	// Allocate vectors in device memory
	float* d_A;
    float* d_B;

    cudaMalloc(&d_A, vectorSize * sizeof(float));
    cudaMalloc(&d_B, vectorSize * sizeof(float));

	// Copy vectors from host memory to device memory
	size_t size = vectorSize * sizeof(float);

	cudaMemcpy(d_A, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, size, cudaMemcpyHostToDevice);

	// Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;

	saxpy_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, scale, vectorSize);

	// Copy result from device memory to host memory
    cudaMemcpy(c, d_B, size, cudaMemcpyDeviceToHost);

	#ifndef DEBUG_PRINT_DISABLE 
		printf(" c = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", c[i]);
		}
		printf(" ... }\n");
	#endif

	int errorCount = verifyVector(a, b, c, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	// Free device memory
    cudaFree(d_A);
    cudaFree(d_B);

	// Free host memory

	// std::cout << "Lazy, you are!\n";
	// std::cout << "Write code, you must\n";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	int i = blockIdx.x * blockDim.x + threadIdx.x; // Get thread ID
	pSums[i] = 0; // Initialize hit for thread to 0

	// Setup RNG
	curandState_t rng;
	curand_init(clock64(), i, 0, &rng);

	for(int j = 0; j < sampleSize; j++) {
		// Get a new random point
		float x = curand_uniform(&rng);
		float y = curand_uniform(&rng);

		if(i < pSumSize) {
			if ( int(x * x + y * y) == 0 ) {
				pSums[i] += 1; // hit
			}
		}
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << std::setprecision(10);
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here

	// On host memory
	uint64_t *pSums;

    pSums = (uint64_t*)malloc(generateThreadCount * sizeof(*pSums));

    if (pSums == NULL) {
        printf("Unable to malloc memory ... Exiting!");
        return -1;
    }

	// Allocate array in device memory
	uint64_t* d_pSums;

    cudaMalloc(&d_pSums, generateThreadCount * sizeof(*d_pSums));

	// Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (generateThreadCount + threadsPerBlock - 1) / threadsPerBlock;


	generatePoints<<<blocksPerGrid, threadsPerBlock>>>(d_pSums, generateThreadCount, sampleSize);

	// Copy result from device memory to host memory
	size_t size = generateThreadCount * sizeof(*pSums);
    cudaMemcpy(pSums, d_pSums, size, cudaMemcpyDeviceToHost);

	// Free device memory
    cudaFree(d_pSums);

	uint64_t hitSum = 0;
	for(int i = 0; i < generateThreadCount; i++) {
		hitSum += pSums[i];
	}

	//	Calculate Pi
	approxPi = ((double)hitSum / (sampleSize * generateThreadCount)); // are we going to have iteration count here?
	approxPi = approxPi * 4.0f;

	#ifndef DEBUG_PRINT_DISABLE 
		printf("hitSum = %d\n", hitSum);
		printf("sampleSize = %d\n", sampleSize);
		printf("threadcount = %d\n", generateThreadCount);
		printf("blocks = %d\n", blocksPerGrid);
	#endif

	// std::cout << "Sneaky, you are ...\n";
	// std::cout << "Compute pi, you must!\n";
	return approxPi;
}
