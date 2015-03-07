/*
 * Copyright (C) 2015 by Julien Delile
 * 
 * This file is part of MECAGEN.
 * 
 * MECAGEN is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation version 3 of the License, or
 * any later version.
 * 
 * MECAGEN is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with MECAGEN.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "vbo.h"

#include "kernel.cu"
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <stdio.h>

// Cuda interface:
void allocateArray(void **devPtr, size_t size)
{ cudaMalloc(devPtr, size); }

void freeArray(void *devPtr)
{ cudaFree(devPtr); }

void copyArrayToDevice(void* device, const void* host, int offset, int size)
{ cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice); }

void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size)
{ cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost); }

// OpenGL interoperability:
void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
{ cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, cudaGraphicsRegisterFlagsNone); }

void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{ cudaGraphicsUnregisterResource(cuda_vbo_resource); }

void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
{
  // gpuErrchk(cudaDeviceSynchronize());
  void *ptr;
  // gpuErrchk(cudaDeviceSynchronize());
  cudaGraphicsMapResources(1, cuda_vbo_resource, 0);
  // gpuErrchk(cudaDeviceSynchronize());
  size_t num_bytes;
  // gpuErrchk(cudaDeviceSynchronize());
  cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes, *cuda_vbo_resource);
  // gpuErrchk(cudaDeviceSynchronize());
  return ptr;
}

void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{ cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0); }


void funcCudaGetLastError(){

	// int deviceCount;
	// cudaGetDeviceCount(&deviceCount);
	// int device;
	// for (device = 0; device < deviceCount; ++device) {
	// cudaDeviceProp deviceProp;
	// cudaGetDeviceProperties(&deviceProp, device);
	// printf("Device %d has compute capability %d.%d.\n",
	// device, deviceProp.major, deviceProp.minor);
	// }

	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf("CUDA error at %s:%i: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));
		exit(-1);
	}
}

void update_vbo(
    mg::DisplayParams *hostDisplayParams,
    float*      outputVBO,            // Where to write on the GPU
    uint        cellNum,                // number of cells
    mg::d3*     cellPosition,                // cell position
    mg::d3*     cellRadius,             // cell radius
    uint*       cellTopologicalNeighbNum,
    uint*       cellTopologicalNeighbId,
    mg::d3*    cellAxisAB,               // Cell axis
    float*      SphereVertex,           // Sphere vertex
    uint*       SphereIndice,           // Sphere indice
    uint        numNeighbMax,
    uint*       SphereVertexNumTriVois, // Voir d'où ça vient
    uint*       SphereVertexIDTriVois,  
    uint*       cellPopulation,
    double*      cellLigand,
    double*      cellProtein,
    mg::f3  colorPop0,
    mg::f3  colorPop1,
    mg::f3  colorPop2,
    mg::f3  colorPop3,
    mg::d3*  cellTriangleNormales,
    uint*       cellType,
    uint*       cellEpiIsPolarized,
    uint*       cellEpiId
    )
{
  
  cudaMemcpyToSymbol<mg::DisplayParams>( displayParams, hostDisplayParams, sizeof(mg::DisplayParams));

  // gpuErrchk(cudaDeviceSynchronize());
  
  update_vbo_old_gpu_1_D<<< cellNum, NUMSPHEREVERTEX >>>(
      outputVBO,            // Where to write on the GPU
        cellNum,                // number of cells
        cellPosition,                // cell position
        cellRadius,             // cell radius
        cellTopologicalNeighbNum,
        cellTopologicalNeighbId,
        cellAxisAB,               // Cell axis
        SphereVertex,           // Sphere vertex
        numNeighbMax,
        cellType,
        cellEpiIsPolarized
    );

  gpuErrchk(cudaDeviceSynchronize());

  update_vbo_old_gpu_2_D<<< cellNum, NUMSPHERETRIANGLE >>>(
      outputVBO,            // Where to write on the GPU
        cellNum,                // number of cells
        SphereIndice,           // Sphere indice
        cellTriangleNormales
    );

  gpuErrchk(cudaDeviceSynchronize());

	update_vbo_old_gpu_3_D<<< cellNum, NUMSPHERETRIANGLE >>>(
			outputVBO,            // Where to write on the GPU
		    cellNum,                // number of cells
    		SphereVertexNumTriVois,
    		SphereVertexIDTriVois,  
        cellPopulation,
    		cellLigand,
    		colorPop0,
    		colorPop1,
    		colorPop2,
    		colorPop3,
        cellTriangleNormales,
        cellProtein,
        cellType,
        cellEpiIsPolarized,
        cellEpiId
		);

	gpuErrchk(cudaDeviceSynchronize());

  // float * temp = new float[10 * NUMSPHEREVERTEX *cellNum];
  // // allocateArray((void**)&temp, 10 * NUMSPHEREVERTEX *cellNum *sizeof(float));
  // copyArrayFromDevice(temp, outputVBO, 0, cellNum2 * 10 * NUMSPHEREVERTEX *sizeof(float));

  // for(uint i=0; i<NUMSPHEREVERTEX *cellNum2;i++){
  //   std::cout << "vertex " << i << " : "  
  //             << temp[10*i+0] << " " 
  //             << temp[10*i+1] << " " 
  //             << temp[10*i+2] << " " 
  //             << temp[10*i+3] << " " 
  //             << temp[10*i+4] << " " 
  //             << temp[10*i+5] << " " 
  //             << temp[10*i+6] << " " 
  //             << temp[10*i+7] << " " 
  //             << temp[10*i+8] << " " 
  //             << temp[10*i+9] << std::endl; 
  // }
  // delete [] temp;
 	
}