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

#include "kernel.cuh"
#include <algorithm>
#include <stdio.h>
#include "../../common/includes/nvVector.h"
#include "define.hpp"
#include "define_gui.h"
#include "thrust_objects.hpp"   //mg::d3
#include "displayparameters.hpp"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void testCudaD()
{
  // printf("testcuda device\n");
  return;
}


__constant__ struct mg::DisplayParams displayParams;

__global__ void update_vbo_old_gpu_1_D(
    float*      outputVBO,            // Where to write on the GPU
    uint        cellNum,                // number of cells
    mg::d3*     cellPosition,                // cell position
    mg::d3*     cellRadius,             // cell radius
    uint*       cellTopologicalNeighbNum,
    uint*       cellTopologicalNeighbId,
    mg::d3*     cellAxisAB,               // Cell axis
    float*      SphereVertex,           // Sphere vertex
    uint        numNeighbMax,
    uint*       cellType,
    uint*       cellEpiIsPolarized
    )
{
  //
  // Thread/cell id:
  //
  uint idcell = blockIdx.x;
  uint idthread = threadIdx.x;

 
  if(idcell >= cellNum){return;}
  if(idthread >= NUMSPHEREVERTEX){return;}
  
  // 
  // Data Initialization:
  //

  mg::d3 n = cellAxisAB[idcell];

  uint numNeigh = cellTopologicalNeighbNum[idcell], idneigh;
  
  if(numNeigh > 30){
    numNeigh = 30;
  }

  float radius_l =  (float)cellRadius[idcell].x, radius_l2;
  float radius_ab = (float)cellRadius[idcell].y;
  float radius_max = radius_l;

  uint celltype = cellType[idcell];
  uint epiPolarized = cellEpiIsPolarized[idcell];

  if(celltype == 2 && epiPolarized == 1){
    radius_max = 1.0f * max(radius_l, radius_ab);
  }
   
  mg::d3 C = cellPosition[idcell];

  mg::d3 newPos(.0f);
  
  newPos.x = C.x + radius_max * SphereVertex[3*idthread + 0];
  newPos.y = C.y + radius_max * SphereVertex[3*idthread + 1];
  newPos.z = C.z + radius_max * SphereVertex[3*idthread + 2];

  
  mg::d3 N, normVect, M;
  
  float lambda;
  float shift, ratio;

  // // Flatten apico-basal borders if epithelial and ab polarized
  // if(celltype == 2 && epiPolarized == 1){

  //   mg::d3 CP = newPos - C;

  //   lambda = dot(CP, n);

  //   if(lambda > radius_ab){
  //     newPos = newPos - (lambda - radius_ab) * n;
  //   }
  //   else if(lambda < - radius_ab){
  //     newPos = newPos + (-lambda- radius_ab ) * n;
  //   }

  // }

  mg::d3 CN, u, v, r;

  for(uint i=0;i<numNeigh; i++){
    
    idneigh = cellTopologicalNeighbId[idcell*NUMNEIGHBTOPOmax + i];
    
    radius_l2 = (float)cellRadius[idneigh].x;
    
    shift = (.5f*radius_l+.5f*radius_l2) * .06f;

    N.x = cellPosition[idneigh].x;
    N.y = cellPosition[idneigh].y;
    N.z = cellPosition[idneigh].z;
    
    CN = N - C; //  vec CN
    r = CN / length(CN);
    
    ratio = radius_l / (radius_l+radius_l2);

    M = ratio * N + (1.0f - ratio) * C;
    
    //u = orthogonale à n passant par centernei
    u = CN - dot(n, CN) * n;
    u /= length(u);

    lambda = 
          dot( 
            newPos - M
            , r)
          /
          dot( u, r) ;
    
    if(lambda > - shift){
      newPos = newPos - (lambda+shift) * u;
    }
  }

  uint idVBO = NUMVBOVAR * (NUMSPHEREVERTEX * idcell + idthread);

  outputVBO[idVBO + 0] = newPos.x;   //x
  outputVBO[idVBO + 1] = newPos.y;   //y
  outputVBO[idVBO + 2] = newPos.z;   //z

}


__global__ void update_vbo_old_gpu_2_D(
    float*      outputVBO,            // Where to write on the GPU
    uint        cellNum,                // number of cells
    uint*       SphereIndice,           // Sphere indice
    mg::d3*  cellTriangleNormales
    )
{

  //
  // Thread/cell id:
  //
  uint idcell = blockIdx.x;
  uint idthread = threadIdx.x;


  if(idcell >= cellNum){return;}
  if(idthread >= NUMSPHERETRIANGLE){return;}
  
  //compute triangle normales  
  uint idvertex = SphereIndice[3 * idthread + 0];
  uint idVBO = NUMVBOVAR * (NUMSPHEREVERTEX * idcell + idvertex);

  mg::d3 a = mg::d3(
            outputVBO[idVBO + 0],
            outputVBO[idVBO + 1],
            outputVBO[idVBO + 2]);
  
  idvertex = SphereIndice[3 * idthread + 1];
  idVBO = NUMVBOVAR * (NUMSPHEREVERTEX * idcell + idvertex);

  mg::d3 b = mg::d3(
            outputVBO[idVBO + 0],
            outputVBO[idVBO + 1],
            outputVBO[idVBO + 2]);
  
  idvertex = SphereIndice[3 * idthread + 2];
  idVBO = NUMVBOVAR * (NUMSPHEREVERTEX * idcell + idvertex);

  mg::d3 c = mg::d3(
            outputVBO[idVBO + 0],
            outputVBO[idVBO + 1],
            outputVBO[idVBO + 2]);
          
  mg::d3 n = cross(b-a, c-a);
  n /= length(n);
  
  cellTriangleNormales[NUMSPHERETRIANGLE * idcell + idthread] = n;
}



__global__ void update_vbo_old_gpu_3_D(
    float*      outputVBO,            // Where to write on the GPU
    uint        cellNum,                // number of cells
    uint*       SphereVertexNumTriVois, // Voir d'où ça vient
    uint*       SphereVertexIDTriVois,  
    uint*       cellPopulation,
    double*      cellLigand,
    mg::f3  colorPop0,
    mg::f3  colorPop1,
    mg::f3  colorPop2,
    mg::f3  colorPop3,
    mg::d3*  cellTriangleNormales,
    double* cellProtein,
    uint*       cellType,
    uint*       cellEpiIsPolarized,
    uint*       cellEpiId
    )
{

  // Thread/cell id:
  //
  uint idcell = blockIdx.x;
  uint idthread = threadIdx.x;


  if(idcell >= cellNum){return;}
  if(idthread >= NUMSPHEREVERTEX){return;}
  
  mg::d3 vertexNormale(.0f);

  uint idtri;
  
  for(uint i= 0; i< SphereVertexNumTriVois[idthread];i++){
    idtri = SphereVertexIDTriVois[10*idthread+i];
    vertexNormale += cellTriangleNormales[NUMSPHERETRIANGLE * idcell + idtri];
  }

  float norm;
  uint idVBO = NUMVBOVAR * (NUMSPHEREVERTEX * idcell + idthread);
  
  norm = length(vertexNormale);

  outputVBO[idVBO + 3] = vertexNormale.x/norm;  //nx
  outputVBO[idVBO + 4] = vertexNormale.y/norm;  //ny
  outputVBO[idVBO + 5] = vertexNormale.z/norm;  //nz


  // Set the colour:
  float hide = .0f;

  mg::f3 color;
  uint cellpop = cellPopulation[idcell];

  if(cellpop == 0){
    color = mg::f3(
              (float)colorPop0.x,
              (float)colorPop0.y,
              (float)colorPop0.z
          );
  }
  else if(cellpop == 1){
    color = mg::f3(
              (float)colorPop1.x,
              (float)colorPop1.y,
              (float)colorPop1.z
          );
  }
  else if(cellpop == 2){
    color = mg::f3(
              (float)colorPop2.x,
              (float)colorPop2.y,
              (float)colorPop2.z
          );
  }
  else if(cellpop == 3){
    color = mg::f3(
              (float)colorPop3.x,
              (float)colorPop3.y,
              (float)colorPop3.z
          ); 
  }

  if(displayParams.currentProtein != -1){
  
    double ratio = cellProtein[idcell*NUMPROTEINmax+displayParams.currentProtein]/displayParams.proteinThreshold[displayParams.currentProtein];
    double nullcolor = .9;    
    color = mg::f3( displayParams.proteinColorR[displayParams.currentProtein] *ratio + (1.0-ratio)*nullcolor, 
                    displayParams.proteinColorG[displayParams.currentProtein] *ratio + (1.0-ratio)*nullcolor,
                    displayParams.proteinColorB[displayParams.currentProtein] *ratio + (1.0-ratio)*nullcolor);
  }
  else if(displayParams.currentLigand != -1){
  
    double ratio = cellLigand[idcell*NUMLIGmax+displayParams.currentLigand]/displayParams.ligandThreshold[displayParams.currentLigand];
    double nullcolor = .9;
    color = mg::f3( displayParams.ligandColorR[displayParams.currentLigand] *ratio + (1.0-ratio)*nullcolor, 
                    displayParams.ligandColorG[displayParams.currentLigand] *ratio + (1.0-ratio)*nullcolor,
                    displayParams.ligandColorB[displayParams.currentLigand] *ratio + (1.0-ratio)*nullcolor);
  }
  else if(displayParams.colorByType == 1){

    uint celltype = cellType[idcell];
    uint epiPolarized = cellEpiIsPolarized[idcell];

    if(celltype == 2 && epiPolarized == 1){
      celltype = 3;
    }

    color = mg::f3( displayParams.typeColorR[celltype], 
                    displayParams.typeColorG[celltype],
                    displayParams.typeColorB[celltype] );

    // //Boundary Formation
    uint cellipid = cellEpiId[idcell];
    if(cellipid == 5){
      color = mg::f3(110.0/255.0, 227.0/255.0, 28.0/255.0);
    }
    else if(cellipid == 8){
      color = mg::f3(255.0/255.0, 66.0/255.0, 0.0/255.0);
    }
  }
  else{
    color = mg::f3(displayParams.noneColorR, displayParams.noneColorG, displayParams.noneColorB);
  }

  outputVBO[idVBO + 6] = color.x;
  outputVBO[idVBO + 7] = color.y;
  outputVBO[idVBO + 8] = color.z;
  outputVBO[idVBO + 9] = hide;
  
}