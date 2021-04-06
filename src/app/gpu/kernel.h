#pragma once

namespace gpu
{
extern "C" void randomiseKernelInit(int numRandomNumbers);
extern "C" void randomiseKernelExec(unsigned int *data, int width, int height);
extern "C" void randomiseKernelCleanup();

extern "C" void executeGameStep(const unsigned int *inputState,
                                unsigned int *outputState, int width,
                                int height);
} // namespace gpu
