#include "kernel.h"

#include <curand.h>
#include <random>

namespace
{
__device__ bool floatToBool(float f) { return f > 0.5f; }

__device__ void toggle(unsigned int *state, int index, bool value)
{
	const unsigned char c = value ? 255 : 0;
	state[index] = c << 24 | c << 16 | c << 8 | c;
}

__device__ bool isToggled(const unsigned int *state, int x, int y, int width,
                          int height)
{
	// wrap coordinates around
	x = (x + width) % width;
	y = (y + height) % height;
	return state[y * width + x] != 0;
}

__device__ int countNeightbours(const unsigned int *state, int x, int y,
                                int width, int height)
{
	// clang-format off
	static int offsets[8][2] = {{-1, -1}, {0, -1}, {1, -1},
	                            {-1,  0},          {1,  0},
	                            {-1, +1}, {0, +1}, {1, +1}};
	// clang-format on
	int n = 0;
	for (int i = 0; i < 8; ++i)
	{
		if (isToggled(state, x + offsets[i][0], y + offsets[i][1], width,
		              height))
		{
			n++;
		}
	}
	return n;
}

__global__ void gameOfLifeKernel(const unsigned int *inputState,
                                 unsigned int *outputState, int arrayWidth,
                                 int arrayHeight)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;
	const int index = y * arrayWidth + x;

	https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life

	bool alive = isToggled(inputState, x, y, arrayWidth, arrayHeight);
	int n = countNeightbours(inputState, x, y, arrayWidth, arrayHeight);

	// Any live cell with two or three live neighbours survives.
	if (alive && (n == 2 || n == 3))
	{
		toggle(outputState, index, true);
	}
	// Any dead cell with three live neighbours becomes a live cell.
	else if (!alive && n == 3)
	{
		toggle(outputState, index, true);
	}
	// All other live cells die in the next generation. Similarly, all other
	// dead cells stay dead.
	else
	{
		toggle(outputState, index, false);
	}
}

namespace random
{
unsigned numRandomNumbers = 0;
float *deviceRandomNumbers = nullptr;
curandGenerator_t rngGenerator;
} // namespace random

__global__ void randomiseValuesKernel(const float *random, int numRandom,
                                      int shuffle, unsigned int *dest,
                                      int arrayWidth)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;

	int index = y * arrayWidth + x;
	const float f = random[(index + shuffle) % numRandom];
	toggle(dest, index, floatToBool(f));
}

} // namespace

namespace gpu
{

extern "C" void randomiseKernelInit(int totalRandomNumbers)
{
	using namespace random;

	randomiseKernelCleanup();
	numRandomNumbers = totalRandomNumbers;
	cudaMalloc((void **)&deviceRandomNumbers, numRandomNumbers * sizeof(float));
	curandCreateGenerator(&rngGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(rngGenerator, 1234ULL);
	curandGenerateUniform(rngGenerator, deviceRandomNumbers, numRandomNumbers);
}

extern "C" void randomiseKernelExec(unsigned int *data, int width, int height)
{
	using namespace random;

	const int sharedMemoryBytes = 0;

	// calculate grid size
	dim3 block(16, 16, 1);
	dim3 grid(width / block.x, height / block.y, 1);

	int shuffle = rand();

	// execute CUDA kernel
	randomiseValuesKernel<<<grid, block, sharedMemoryBytes>>>(
	    deviceRandomNumbers, numRandomNumbers, shuffle, data, width);

	cudaDeviceSynchronize();
}

extern "C" void randomiseKernelCleanup()
{
	using namespace random;

	if (deviceRandomNumbers != nullptr)
	{
		cudaFree(deviceRandomNumbers);
		deviceRandomNumbers = nullptr;

		curandDestroyGenerator(rngGenerator);
	}
}

extern "C" void executeGameStep(const unsigned int *inputState,
                                unsigned int *outputState, int width,
                                int height)
{
	const int sharedMemoryBytes = 0;

	// calculate grid size
	dim3 block(16, 16, 1);
	dim3 grid(width / block.x, height / block.y, 1);

	// execute CUDA kernel
	gameOfLifeKernel<<<grid, block, sharedMemoryBytes>>>(
	    inputState, outputState, width, height);

	cudaDeviceSynchronize();
}

} // namespace gpu
