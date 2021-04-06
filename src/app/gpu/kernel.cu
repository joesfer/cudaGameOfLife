#include "kernel.h"

namespace
{
// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b)
{
	r = clamp(r, 0.0f, 255.0f);
	g = clamp(g, 0.0f, 255.0f);
	b = clamp(b, 0.0f, 255.0f);
	return (int(b) << 16) | (int(g) << 8) | int(r);
}

__global__ void cudaProcess(unsigned int *data, int arrayWidth)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;

	uchar4 c4 = make_uchar4((x & 0x20) ? 100 : 0, 0, (y & 0x20) ? 100 : 0, 0);
	data[y * arrayWidth + x] = rgbToInt(c4.z, c4.y, c4.x);
}

} // namespace

namespace gpu
{

extern "C" void dispatchKernel(dim3 grid, dim3 block, int sbytes,
                               unsigned int *data, int arrayWidth)
{
	cudaProcess<<<grid, block, sbytes>>>(data, arrayWidth);
}

} // namespace gpu
