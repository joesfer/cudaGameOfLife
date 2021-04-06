#include <cstdio>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl2.h"

#include "defines.h"
#include "gpu/info.h"
#include "gpu/kernel.h"

#include "app.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

Application::~Application()
{
	cleanupCuda();
	cleanupGL();
}

bool Application::init() { return initGL() && initCuda(); }

void Application::mainLoop()
{

	// Main loop
	while (!glfwWindowShouldClose(m_window))
	{
		compute();
		drawFrame();
	}
}

bool Application::initGL()
{
	// Create resources

	if (!glfwInit())
	{
		return false;
	}

	m_window = glfwCreateWindow(1024, 768, "Test", NULL, NULL);
	if (!m_window)
	{
		return false;
	}

	glfwMakeContextCurrent(m_window);
	glfwSwapInterval(1); // Enable vsync

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(m_window, true);
	ImGui_ImplOpenGL2_Init();

	// Cache display dimensions
	// TODO handle window resizing?
	glfwGetFramebufferSize(m_window, &m_displayWidth, &m_displayHeight);

	// Create a slightly lower resolution buffer to see the cells better
	m_bufferWidth = m_displayWidth / 4;
	m_bufferHeight = m_displayHeight / 4;

	// Create texture for CUDA and GL to interop
	// https://github.com/lxc-xx/CudaSample/blob/master/NVIDIA_CUDA-5.5_Samples/3_Imaging/simpleCUDA2GL/main.cpp
	glGenTextures(1, &m_cudaTexture);
	glBindTexture(GL_TEXTURE_2D, m_cudaTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_bufferWidth, m_bufferHeight, 0,
	             GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	return true;
}

bool Application::initCuda()
{
	gpu::info();

	gpu::randomiseKernelInit(m_bufferWidth * m_bufferHeight);
	for (int i = 0; i < 2; ++i)
	{
		m_cudaBuffer[i] =
		    static_cast<unsigned int *>(ALLOCATE(getTextureDataSize()));

		gpu::randomiseKernelExec(m_cudaBuffer[i], m_bufferWidth,
		                         m_bufferHeight);
	}

	gpu::randomiseKernelCleanup();

	cudaGraphicsGLRegisterImage(&m_cudaTextureResource, m_cudaTexture,
	                            GL_TEXTURE_2D,
	                            cudaGraphicsMapFlagsWriteDiscard);
	return true;
}

void Application::compute()
{
	const unsigned sourceBuffer = m_currentBuffer;
	const unsigned destBuffer = (sourceBuffer + 1) % 2;
	m_currentBuffer = destBuffer;

	{
		// run the Cuda kernel
		gpu::executeGameStep(m_cudaBuffer[sourceBuffer],
		                     m_cudaBuffer[destBuffer], m_bufferWidth,
		                     m_bufferHeight);
	}

	{ // copy to OpenGL
		// We want to copy m_cudaBuffer data to the texture
		// map buffer objects to get CUDA device pointers
		cudaArray *texturePtr;
		cudaGraphicsMapResources(1, &m_cudaTextureResource, 0);
		cudaGraphicsSubResourceGetMappedArray(&texturePtr,
		                                      m_cudaTextureResource, 0, 0);

		const size_t pixelSize = sizeof(GLubyte) * 4;
		const size_t widthBytes = pixelSize * m_bufferWidth;
		const size_t srcPitchBytes = pixelSize * m_bufferWidth;
		const size_t heightRows = m_bufferHeight;
		cudaMemcpy2DToArray(texturePtr,               // dst
		                    0,                        // wOffset
		                    0,                        // hOffset
		                    m_cudaBuffer[destBuffer], // src
		                    srcPitchBytes,            // source picth
		                    widthBytes,               // width
		                    heightRows,               // height,
		                    cudaMemcpyDefault);

		cudaGraphicsUnmapResources(1, &m_cudaTextureResource, 0);
	}
}

void Application::drawFrame()
{
	glfwPollEvents();

	// Start the Dear ImGui frame
	ImGui_ImplOpenGL2_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	{ // Show some stats
		ImGui::Begin("Stats");
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
		            1000.0f / ImGui::GetIO().Framerate,
		            ImGui::GetIO().Framerate);
		ImGui::End();
	}

	// Rendering

	glViewport(0, 0, m_displayWidth, m_displayHeight);
	glClear(GL_COLOR_BUFFER_BIT);

	{ // draw texture

		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glViewport(0, 0, m_displayWidth, m_displayHeight);

		glBindTexture(GL_TEXTURE_2D, m_cudaTexture);
		glEnable(GL_TEXTURE_2D);

		glBegin(GL_QUADS);
		glTexCoord2f(0.0, 0.0);
		glVertex3f(-1.0, -1.0, 0.5);
		glTexCoord2f(1.0, 0.0);
		glVertex3f(1.0, -1.0, 0.5);
		glTexCoord2f(1.0, 1.0);
		glVertex3f(1.0, 1.0, 0.5);
		glTexCoord2f(0.0, 1.0);
		glVertex3f(-1.0, 1.0, 0.5);
		glEnd();

		glMatrixMode(GL_PROJECTION);
		glPopMatrix();

		glDisable(GL_TEXTURE_2D);
	}

	ImGui::Render();

	ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

	glfwMakeContextCurrent(m_window);
	glfwSwapBuffers(m_window);
}

void Application::cleanupCuda()
{
	for (int i = 0; i < 2; ++i)
	{
		FREE(m_cudaBuffer[i]);
	}
	cudaDeviceReset();
}

void Application::cleanupGL()
{
	glfwDestroyWindow(m_window);
	glfwTerminate();
}

size_t Application::getTextureDataSize() const
{
	int numTexels = m_bufferWidth * m_bufferHeight;
	int numValues = numTexels * 4;
	int texDataBytes = sizeof(GLubyte) * numValues;
	return texDataBytes;
}
