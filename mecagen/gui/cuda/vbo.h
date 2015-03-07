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

#ifndef CUDAVBO_H_
#define CUDAVBO_H_

#include "thrust_objects.hpp"   //d3
#include "define.hpp"   //d3
#include "displayparameters.hpp"   //d3


// Cuda interface:
void allocateArray(void **devPtr, size_t size);
void freeArray(void *devPtr);
void copyArrayToDevice(void* device, const void* host, int offset, int size);
void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);

// OpenGL interoperability:
void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);

// Cuda computation:

void funcCudaGetLastError();
void testCuda();

void update_vbo(
    mg::DisplayParams *hostDisplayParams,
    float*      outputVBO,            // Where to write on the GPU
    uint        cellNum,                // number of cells
    mg::d3*     	cellPosition,                // cell position
    mg::d3*     cellRadius,             // cell radius
    uint*       cellTopologicalNeighbNum,
    uint*       cellTopologicalNeighbId,
    mg::d3*    		cellAxisAB,               // Cell axis
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
    mg::d3* cellTriangleNormales,
    uint*       cellType,
    uint*       cellEpiIsPolarized,
    uint*       cellEpiId
    );

#endif
