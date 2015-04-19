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

#include "metaparam.hpp"
#include "state_host.hpp"
#include "param_host.hpp"
#include "thrust_objects.hpp"
#include "serialization.hpp"
#include "custom_objects.hpp"

#include <stdio.h>
#include <set>

using namespace mg;

int main( int argc, const char* argv[] ){
  
  /**********************************/
  /**********************************/
  /*******    MetaParam     *********/
  /**********************************/
  /**********************************/
  
  MetaParam<HOST>     metaParam;
  
  // Maximum cell number allowed
  metaParam.numCellsMax[0] = 5000;

  metaParam.spatialBorderMin[0] = d3(-100,-100,-80);
  metaParam.spatialBorderMax[0] = d3(100,100,80);

  metaParam.grid_SizeMax[0]     = 64;


  /*** Embryo axes ***/
  metaParam.embryoAxes[0] = d3(-715.0-446.0, -3461.0-511.0, -385.0-481.5);
  metaParam.embryoAxes[0] /= length(metaParam.embryoAxes[0]);
  metaParam.embryoAxes[1] = d3(-metaParam.embryoAxes[0].y,metaParam.embryoAxes[0].x,0);
  metaParam.embryoAxes[1] /= length(metaParam.embryoAxes[1]);
  metaParam.embryoAxes[2] = cross(metaParam.embryoAxes[0], metaParam.embryoAxes[1]);
  metaParam.embryoAxes[2] /= length(metaParam.embryoAxes[2]);

  save< MetaParam<HOST> >(metaParam, "metaparam_archive.xml");

  /**********************************/
  /**********************************/
  /**********   State       *********/
  /**********************************/

  // load BoxTissueFrom1cell_grn_default state archive
  State_Host state(&metaParam);
 /*** Cell number ***/
  state.numCells[0] = 1548;
  /*** Timestep ***/
  state.currentTimeStep[0] = 0;   


  double Ryolk = 1.0; //100.0;//.51;//100.0;

  // state.cellPosition[0] = d3(.0);
  printf("Loading cell positions...   \n");
  char string[300];
  FILE *file;
  sprintf(string, "%s","./CellStartPos");
  file = fopen(string, "r");
  uint temp;
  if(file==NULL) {
    printf("Error: can't open file. CellStartPos\n");
  }
  else {
    uint i = 0 ;   
    double x,y,z; 
    while(i<state.numCells[0]){  
      temp = fscanf(file, "%lf", &x);
      temp = fscanf(file, "%lf", &y);
      temp = fscanf(file, "%lf", &z);
      state.cellPosition[i] = d3(Ryolk*x, Ryolk*y, Ryolk*z);
      i++;
    }      
    fclose(file);
  }

  /*** Embryo center ***/
  state.embryoCenter[0] = d3(.0);
  for(int i=0; i<state.numCells[0]; i++){
    state.embryoCenter[0] += state.cellPosition[i];
  }
  state.embryoCenter[0] /= (double)(state.numCells[0]);

  /*** Cell radii ***/
  for(int i=0; i<state.numCells[0]; i++){
    state.cellRadius[i] = d3(Ryolk*0.030824, 0,0);
  }

  /*** Cell cycle ***/
  std::default_random_engine generator(0);
  std::normal_distribution<double> distribution(1000.0, 1000.0);
  // std::normal_distribution<double> distribution(500.0, 100.0);

  for(uint i=0; i<state.numCells[0]; i++){
    state.cellState[i] = 0;
    state.cellTimer[i] = 0;
    state.cellGeneration[i] = 0;
    int ccl = (int)(distribution(generator));
    while(ccl<=0){
      // printf("Error lifetime less than 0 : cell %d   lifetime %d\n", i,ccl);
      ccl = (int)(distribution(generator));
    }
    assert(ccl > 0);
    // state.cellCycleLength[i] = 1000000;//(uint)ccl;
    state.cellCycleLength[i] = (uint)ccl;
    
  }

  /*** Cell binary Id ***/
  for(int i=0; i<state.numCells[0]; i++){
    state.cellId_bits[i] = 0;   //TODO: give unique binary id to each cell
  }

  /*** Cell Ligands ***/
  for(int i=0; i<state.numCells[0] * NUMLIGmax; i++){
    state.cellLigand[i] = 0.0;

    // if(i==0){   // Ubiquitous
    //   state.cellLigand[i] = 66.667;
    // }
  }

  /*** Cell protein ****/
  for(int i=0; i<state.numCells[0] * NUMPROTEINmax; i++){
    state.cellProtein[i] = 0.0;
  }    
 

  // Initialize yolk particles' positions

  printf("Loading yolk interior particles...   \n");
  sprintf(string, "%s","./IYolkStartPos2");
  file = fopen(string, "r");
    if(file==NULL) {
    printf("Error: can't open file. IYolkStartPos\n");
  }
  else {
    uint i = 0 ;   
    double x,y,z; 
    while(i<NUMPARTYOLKINTERIOR){  
      temp = fscanf(file, "%lf", &x);
      temp = fscanf(file, "%lf", &y);
      temp = fscanf(file, "%lf", &z);
      state.customState.yolkPosition[i+NUMPARTYOLKMEMBRANE] = d3(Ryolk*x, Ryolk*y, Ryolk*z);
      i++;
    }      
    fclose(file);
  }

  printf("Computing yolk interior radius...\n");
  // The spherical yolk (radius Ryolk) is filled with NUMPARTYOLKINTERIOR yolk interior particle of volume 4/3 * pi * YolkPartRadius^3
  // i.e. NUMPARTYOLKINTERIOR * 4/3 * pi * YolkPartRadius^3 = 4*3*pi*(Ryolk - YolkPartRadius)^3
  // We deduce
  state.customState.yolkInteriorRadius[0]  = .51 * Ryolk / ( 1.0 + pow((double)NUMPARTYOLKINTERIOR, 1.0/3.0) );
  state.customState.yolkInteriorDistMax[0] = 2.0 * 1.2414 * state.customState.yolkInteriorRadius[0];
  state.customState.yolkInteriorDistEq[0]  = 2.0 * 0.9523128 * state.customState.yolkInteriorRadius[0];
  // state.customState.yolkInteriorDistEq[0]  = 2.0 * 0.9523128 * state.customState.yolkInteriorRadius[0];
  
  printf("Loading yolk membrane particles...   \n");
  sprintf(string, "%s","./MemYolkStartPos2");    
  file = fopen(string, "r");
  if(file==NULL) {
    printf("Error: can't open file. %s\n",string);
  }
  else {      
    uint i = 0 ;   
    double x,y,z; 
    while(i<NUMPARTYOLKMEMBRANE){  
      temp = fscanf(file, "%lf", &x);
      temp = fscanf(file, "%lf", &y);
      temp = fscanf(file, "%lf", &z);
      state.customState.yolkPosition[i] = d3(Ryolk*x, Ryolk*y, Ryolk*z);
      i++;
    }
    fclose(file);
  }

  printf("Loading yolk membrane rank 1 neighbor ids...   \n");  
  sprintf(string, "%s","./yolkNeighb2562.txt");
  file = fopen(string, "r");
  char useless[100];
  uint uselessd; 
  if(file==NULL) {
    printf("Error: can't open file. yolkNeighb2562\n");
  }
  else {
    uint i = 0 ;   
    while(i<NUMPARTYOLKMEMBRANE){ 
      temp = fscanf(file, "%s", useless);   //"cell"
      temp = fscanf(file, "%d", &uselessd);   //idcell
      temp = fscanf(file, "%s", useless);   //"nv"
      temp = fscanf(file, "%d", &(state.customState.yolkMembraneNeighbNum[i]));
      temp = fscanf(file, "%s", useless);  // "listt"
      for(uint j=0;j<state.customState.yolkMembraneNeighbNum[i];j++){
        temp = fscanf(file, "%d", &(state.customState.yolkMembraneNeighbId[NUMNEIGHMYMYmax*i+j]));
      }
      i++;
    }
    fclose(file);
  }

  printf("Deducing yolk membrane rank 2 neighbor ids...   \n");  
  uint idneigh1, idneigh2;
  std::set<uint> rank2;
  std::set<uint>::iterator it_rank2;
  for(uint i=0; i<NUMPARTYOLKMEMBRANE; i++){
    rank2.clear();
    for(uint j=0; j<state.customState.yolkMembraneNeighbNum[i]; j++){
      idneigh1 = state.customState.yolkMembraneNeighbId[NUMNEIGHMYMYmax*i+j];
      for(uint k=0; k<state.customState.yolkMembraneNeighbNum[idneigh1]; k++){
        idneigh2 = state.customState.yolkMembraneNeighbId[NUMNEIGHMYMYmax*idneigh1+k];
        rank2.insert(idneigh2);
      }
    }
    rank2.erase(i);
    for(uint j=0; j<state.customState.yolkMembraneNeighbNum[i]; j++){
      idneigh1 = state.customState.yolkMembraneNeighbId[NUMNEIGHMYMYmax*i+j];
      rank2.erase(idneigh1);    //rank2 contains only rank2 neighb
    }
    state.customState.yolkMembraneNeighbNum[i + NUMPARTYOLKMEMBRANE] = state.customState.yolkMembraneNeighbNum[i] + (uint)rank2.size();
    uint j=0;
    for(it_rank2=rank2.begin(); it_rank2!=rank2.end(); it_rank2++){
      state.customState.yolkMembraneNeighbId[
          NUMNEIGHMYMYmax*i+ state.customState.yolkMembraneNeighbNum[i] + j
            ] = *it_rank2;
      j++;
    }
  }

  printf("Computing yolk membrane rest length from a spherical yolk...   \n");
  d3 * partMemYolkPos = new d3[NUMPARTYOLKMEMBRANE];
  sprintf(string, "%s","./MemYolkStartPos_perfectsphere");
  file = fopen(string, "r");
  if(file==NULL) {
    printf("Error: can't open file. %s\n",string);
  }
  else {
    uint i = 0 ;   
    double x,y,z; 
    while(i<NUMPARTYOLKMEMBRANE){  
      temp = fscanf(file, "%lf", &x);
      temp = fscanf(file, "%lf", &y);
      temp = fscanf(file, "%lf", &z);
      partMemYolkPos[i] = d3(x,y,z);
      i++;
    }
    fclose(file);
  }

  d3 yolkCenter_sphere = d3(.0);
  // d3 yolkCenter = d3(.0);
  
  for(uint i=0;i<NUMPARTYOLKMEMBRANE;i++){
    // yolkCenter += state.customState.yolkMembranePosition[i];
    yolkCenter_sphere += partMemYolkPos[i];
  }
  
  // yolkCenter/=(double)NUMPARTYOLKMEMBRANE;
  yolkCenter_sphere /= (double)NUMPARTYOLKMEMBRANE;
  
  // //center sphere
  // for(uint i=0;i<NUMPARTYOLKMEMBRANE;i++){
  //   partMemYolkPos[i] -= yolkCenter_sphere;
  // }

  for(uint i=0;i<NUMPARTYOLKMEMBRANE;i++){
    d3 posmemyolk = partMemYolkPos[i];
    double radius = length(posmemyolk - yolkCenter_sphere);
    partMemYolkPos[i] = yolkCenter_sphere + .51 * Ryolk * (posmemyolk - yolkCenter_sphere) / radius;
  }
  
  double rl_averg = .0, numlink = .0;
  for(uint i=0;i<NUMPARTYOLKMEMBRANE;i++){
    for(uint j=0;j<state.customState.yolkMembraneNeighbNum[i+NUMPARTYOLKMEMBRANE];j++){
      uint idneigh = state.customState.yolkMembraneNeighbId[NUMNEIGHMYMYmax*i + j];
      state.customState.yolkMembraneNeighbRL[NUMNEIGHMYMYmax*i + j] = length(partMemYolkPos[i] - partMemYolkPos[idneigh]);
      // printf("A. part %d neighb %d id %d -> rl %lf\n",
      //     i, j, idneigh, state.customState.yolkMembraneNeighbRL[NUMNEIGHMYMYmax*i + j]);
      if( j < state.customState.yolkMembraneNeighbNum[i] ){
        rl_averg += .95*state.customState.yolkMembraneNeighbRL[NUMNEIGHMYMYmax*i + j];
        numlink++;
      }
    }  
  }
  rl_averg /= numlink;  
  
  // for(int i = 0; i < NUMPARTYOLKMEMBRANE; i++){
  //   std::cout 
  //             << "part " << i 
  //             << " / numNeighb1 " << state.customState.yolkMembraneNeighbNum[i] 
  //             << " : ";
  //   for(uint j=0; j < state.customState.yolkMembraneNeighbNum[i]; j++){
  //     std::cout << state.customState.yolkMembraneNeighbId[NUMNEIGHMYMYmax*i+j] << " ";
  //   }
  //   std::cout << " / numNeighb2 " << state.customState.yolkMembraneNeighbNum[i+NUMPARTYOLKMEMBRANE] 
  //             << " : ";
  //   for(uint j=0; j < state.customState.yolkMembraneNeighbNum[i+NUMPARTYOLKMEMBRANE]; j++){
  //     std::cout << state.customState.yolkMembraneNeighbId[NUMNEIGHMYMYmax*i+j] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  //sort neighbors to obtain normal easily
  double* angletab = new double[NUMNEIGHMYMYmax*NUMPARTYOLKMEMBRANE*sizeof(double)];
  double* rltemp = new double[NUMNEIGHMYMYmax*NUMPARTYOLKMEMBRANE*sizeof(double)];
  uint* idtemp = new uint[NUMNEIGHMYMYmax*NUMPARTYOLKMEMBRANE*sizeof(uint)];
  
  for(uint i=0;i<NUMPARTYOLKMEMBRANE;i++){

    d3 MY = partMemYolkPos[i];
    d3 normal = MY - yolkCenter_sphere;
    normal /= length(normal);
    
    d3 normal2 = d3(-normal.y,normal.x,.0);
    normal2 /= length(normal2);
    
    d3 normal3 = cross(normal, normal2);
    
    //working on rank1 only
    for(uint j=0;j<state.customState.yolkMembraneNeighbNum[i];j++){
      
      uint idvois = state.customState.yolkMembraneNeighbId[NUMNEIGHMYMYmax*i + j];
      double rl = state.customState.yolkMembraneNeighbRL[NUMNEIGHMYMYmax*i + j];
      d3 relPos = partMemYolkPos[idvois] - MY;
      relPos /= length(relPos);
      
      //projection on n2 and n3
      double p2 = dot(normal2, relPos);
      double p3 = dot(normal3, relPos);
      
      double angle;
      if( p3 > 0 ){
        angle = acos( p2 / (sqrt(p2*p2+p3*p3)+.00001) );
      }
      else{
        angle = 2 * PI - acos( p2 / (sqrt(p2*p2+p3*p3)+.00001) );
      }
      //insert vois at the ordered position
      uint insertPos = 0;
      for(uint k=0;k<j;k++){
        if( angletab[i*NUMNEIGHMYMYmax+k] > angle ){
          break;
        }
        insertPos = k+1;
      }
      
      for(uint k=j;k>insertPos;k--){
        angletab[i*NUMNEIGHMYMYmax + k ] = angletab[i*NUMNEIGHMYMYmax + k -1];
        idtemp[i*NUMNEIGHMYMYmax + k ] = idtemp[i*NUMNEIGHMYMYmax + k -1];
        rltemp[i*NUMNEIGHMYMYmax + k ] = rltemp[i*NUMNEIGHMYMYmax + k -1];
      }
      
      angletab[i*NUMNEIGHMYMYmax + insertPos ] = angle;
      idtemp[i*NUMNEIGHMYMYmax + insertPos ] = idvois;
      rltemp[i*NUMNEIGHMYMYmax + insertPos ] = rl;
    }
  }

  for(uint i=0;i<NUMPARTYOLKMEMBRANE;i++){
    for(uint j=0;j<state.customState.yolkMembraneNeighbNum[i];j++){
      state.customState.yolkMembraneNeighbId[NUMNEIGHMYMYmax*i + j] = idtemp[i*NUMNEIGHMYMYmax + j];
      state.customState.yolkMembraneNeighbRL[NUMNEIGHMYMYmax*i + j] = rltemp[i*NUMNEIGHMYMYmax + j];
      // printf("B. part %d neighb %d id %d -> rl %lf\n",
      //     i, j, idtemp[i*NUMNEIGHMYMYmax + j], rltemp[i*NUMNEIGHMYMYmax + j]);
    }
  }

  printf("Computing yolk membrane radius...\n");
  // Yolk membrane particle are considered as hexagonal prism
  // whose length is 2 * yolkInteriorRadius 
  // and altitude is half the average rank1 neighbor length 
  // The hexagon surface is:
  double hexagon_surface = 2 * std::sqrt(3.0) * rl_averg * rl_averg;
  // The volume of the yolk membrane particle is::
  double ym_volume = 2 * state.customState.yolkInteriorRadius[0] * hexagon_surface;
  // However, the yolk particle surfaces are calculated by considering them as spheroidal.
  // We store a single yolk membrane radius calculated from the sphere equation
  state.customState.yolkMembraneRadius[0]  = std::pow( (double)(3.0 * ym_volume / (4.0 * PI)), (double)(1.0/3.0));
  
  // printf("radius ypart %lf  mysphere %lf rl_aver %lf volumeMY %lf auei %lf tiers %lf ratio %lf\n", 
  //     , radius_spherical, rl_averg,volumeMY, 3 * volumeMY / (4*PI),
  //     (double)(1.0/3.0), m_params_const.YolkPartRadius/radius_spherical );

  printf("Loading evl particles position...   \n");
  state.customState.numPartEVL[0] = 250;

  sprintf(string, "%s","./EVLStartPos");
  file = fopen(string, "r");
    if(file==NULL) {
    printf("Error: can't open file. EVLStartPos\n");
  }
  else {
    uint i = 0 ;   
    double x,y,z; 
    while(i<state.customState.numPartEVL[0]){  
      temp = fscanf(file, "%lf", &x);
      temp = fscanf(file, "%lf", &y);
      temp = fscanf(file, "%lf", &z);
      state.customState.evlPosition[i] = d3(Ryolk*x, Ryolk*y, Ryolk*z);
      // state.customState.evlRadius[i].x = Ryolk*0.038836;
      // state.customState.evlRadius[i].y = Ryolk*0.038836;
      i++;
    }      
    fclose(file);
  }

  sprintf(string, "%s","./EVLStartRadiusLat");
  file = fopen(string, "r");
    if(file==NULL) {
    printf("Error: can't open file. EVLStartRadiusLat\n");
  }
  else {
    uint i = 0 ;   
    double rad; 
    while(i<state.customState.numPartEVL[0]){  
      temp = fscanf(file, "%lf", &rad);
      state.customState.evlRadius[i].x = Ryolk*rad;
      state.customState.evlRadius[i].y = Ryolk*0.038836;
      i++;
    }      
    fclose(file);
  }
  
  printf("Loading evl particles normales...   \n");
  sprintf(string, "%s","./EVLStartNormal");
  file = fopen(string, "r");
    if(file==NULL) {
    printf("Error: can't open file. EVLStartNormal\n");
  }
  else {
    uint i = 0 ;   
    double x,y,z; 
    while(i<state.customState.numPartEVL[0]){  
      temp = fscanf(file, "%lf", &x);
      temp = fscanf(file, "%lf", &y);
      temp = fscanf(file, "%lf", &z);
      state.customState.evlNormal[i] = d3(x, y, z);
      state.customState.evlNormal[i] /= length(state.customState.evlNormal[i]);
      i++;
    }      
    fclose(file);
  }

  printf("Initializing evl timer...\n");
  for(uint i=0; i<NUMPARTEVLmax; i++){
    state.customState.evlTimer[i] = 0;
  }
  
  printf("Loading yolk activation...   \n");
  sprintf(string, "%s","./MemYolkStartActivated");
  file = fopen(string, "r");
    if(file==NULL) {
    printf("Error: can't open file. MemYolkStartActivated\n");
  }
  else {
    uint i = 0, act;
    while(i<NUMPARTYOLKMEMBRANE){  
      temp = fscanf(file, "%d", &act);
      state.customState.yolkMembraneActivated[i] = act;
      i++;
    }      
    fclose(file);
  }
  
  printf("Loading yolk eYSL...   \n");
  sprintf(string, "%s","./MemYolkStartEYSL");
  file = fopen(string, "r");
    if(file==NULL) {
    printf("Error: can't open file. MemYolkStartEYSL\n");
  }
  else {
    uint i = 0, eysl;
    while(i<NUMPARTYOLKMEMBRANE){  
      temp = fscanf(file, "%d", &eysl);
      state.customState.yolkMembraneEYSL[i] = eysl;
      i++;
    }      
    fclose(file);
  }
  
  printf("Computing yolk membrane tangent parameters... \n");
  //Get Animal and Vegatal pole indices
  double scalmax = -1.0;
  double scalmin = 1.0;
  uint imin = 0,imax = 0;
  for(uint i=0; i<NUMPARTYOLKMEMBRANE; i++){
    double scal = dot(partMemYolkPos[i], metaParam.embryoAxes[0]);
    if(scal < scalmin){
      scalmin = scal;
      imin = i;
    }
    if(scal > scalmax){
      scalmax = scal;
      imax = i;
    }
  }
  d3 An   = partMemYolkPos[imax];
  d3 V    = partMemYolkPos[imin];

  uint ineigh, idneigh;
  for(uint i=0; i<NUMPARTYOLKMEMBRANE; i++){

    d3 normal(.0);
    
    d3 A = partMemYolkPos[i];
    d3 C = partMemYolkPos[ state.customState.yolkMembraneNeighbId[i*NUMNEIGHMYMYmax+0] ];
    d3 AC = C - A;
    AC /= length(AC);
    
    for (uint k = 1; k < state.customState.yolkMembraneNeighbNum[i]; k++)
    { 
      ineigh = NUMNEIGHMYMYmax*i + k;
      idneigh = state.customState.yolkMembraneNeighbId[ ineigh ];
      
      d3 B = C;
      d3 AB = B - A;
      AB /= length(AB);
      
      C = partMemYolkPos[idneigh];
      AC = C - A;
      AC /= length(AC);
      
      d3 normaltemp = cross(AB, AC);
      normaltemp /= length(normaltemp);
      normal += normaltemp;
    }
    
    d3 B=C;
    d3 AB = B - A;
    AB /= length(AB);
    
    ineigh = NUMNEIGHMYMYmax*i;
    idneigh = state.customState.yolkMembraneNeighbId[ ineigh ];
    C = partMemYolkPos[idneigh];
    AC = C - A;
    AC /= length(AC);
    
    d3 normaltemp = cross(AB, AC);
    normaltemp /= length(normaltemp);
    normal += normaltemp;
    
    normal /= length(normal);
    
    d3 AV = V - A;
    d3 AAn = An - A;
    
    d3 P = cross(AV, AAn);

    d3 T;
    T.x = 1.0;
    T.y = - T.x * ( normal.x * P.z - P.x * normal.z ) / ( normal.y * P.z - P.y * normal.z );
    T.z = - 1.0 / normal.z * ( T.x*normal.x + T.y*normal.y );
    T /= length(T);
    
    if( dot(An - V,T) > 0){
      T *= -1.0;
    }
    
    //selectionne les deux voisines le plus proches
    double min1 = -100.0, min2 = -120.0;
    uint imin1 = 0, imin2 = 0;
    
    for (uint k = 0; k < state.customState.yolkMembraneNeighbNum[i]; k++){
      
      ineigh = NUMNEIGHMYMYmax*i + k;
      idneigh = state.customState.yolkMembraneNeighbId[ ineigh ];
      
      B = partMemYolkPos[idneigh];
      
      AB = B - A;
      AB /= length(AB);
      
      double scal = dot(AB, T);
      
      if(scal>min1){
        min2 = min1;
        imin2 = imin1;
        min1 = scal;
        imin1 = idneigh;
      }
      else if(scal>min2){
        min2 = scal;
        imin2 = idneigh;
      }
    }
    
    //decompose T sur les deux vois les plus proche : T = coeff1 *AB + coeff2*AC
    //pour y arriver, on decompose AB et AC dans une base orthogonale (e1, e2) 
    AB = partMemYolkPos[imin1] - A;
    AC = partMemYolkPos[imin2] - A;
    
    d3 e1 = AB / length(AB);
    d3 e3 = cross(AB, AC);
    e3 /= length(e3);
    d3 e2 = cross(e1, e3);
    
    double coeff1 = ( dot(T,e1)*dot(AC,e2) - dot(T,e2)*dot(AC,e1) ) / ( dot(AB,e1)*dot(AC,e2) - dot(AB,e2)*dot(AC,e1) );
    double coeff2 = ( dot(T,e1)*dot(AB,e2) - dot(T,e2)*dot(AB,e1) ) / ( dot(AC,e1)*dot(AB,e2) - dot(AC,e2)*dot(AB,e1) );

    if(i==imin || i==imax){ //animal pole or vegetal pole particle
      state.customState.yolkMembraneTangentParams[i] = d4(0, 0, 0, 0);     
    }
    else{
      state.customState.yolkMembraneTangentParams[i] = d4(imin1, coeff1, imin2, coeff2);     
    }
  }



  // Associate yolk membrane id with their rank from margin
  uint * PartMYonLinefromLineID  = new uint[300 * 100]; //300 lignes max avec 100 part/ligne max
  memset(PartMYonLinefromLineID, 0, 300*100*sizeof(uint));    
    
  uint *PartMYonLinefromLineIDnum  = new uint[300]; 
  memset(PartMYonLinefromLineIDnum, 0, 300*sizeof(uint)); 
  
  uint lineid = 0;
  for(uint i=0; i<NUMPARTYOLKMEMBRANE; i++){
    uint margin = 0;
    if(state.customState.yolkMembraneActivated[i] == 1){
      for(uint j=0; j < state.customState.yolkMembraneNeighbNum[i]; j++){
        if( state.customState.yolkMembraneActivated[ state.customState.yolkMembraneNeighbId[NUMNEIGHMYMYmax*i+j] ] == 0 ){
          margin = 1;
        }
      }
    }
    state.customState.yolkMembraneEYSL[i] = margin;
    if(margin == 1){
      PartMYonLinefromLineID[lineid * 100 + 0] = i;
      PartMYonLinefromLineIDnum[lineid] = 1;
      lineid++;
    }
  }
  uint numPartMYmargin0 = lineid;
  
  double * PartMYrankAV = new double[NUMPARTYOLKMEMBRANE];
  for(uint i=0; i<NUMPARTYOLKMEMBRANE; i++){
    PartMYrankAV[i] = length( partMemYolkPos[i] - V);
  }

  d3 AnV = An-V;
  AnV /= length(AnV);
  
  d3 normal1 = d3(-AnV.y,AnV.x,.0);
  normal1 /= length(normal1);
    
  d3 normal2 = cross(AnV, normal1);
  
  scalmin = cos ( .5 * 3.0 * PI / (double)numPartMYmargin0); //3.0 au lieu de 2 pour s'assurer que toutes les MY sont au moins dans une ligne
  
  for(uint i=0;i<NUMPARTYOLKMEMBRANE;i++){
    
    //si on est hors de iYSL
    if(state.customState.yolkMembraneActivated[i] == 0){
      
      d3 PA = partMemYolkPos[i] - An;
      PA /= length(PA);

      //parcours toutes les lignes
      for(uint j=0;j<numPartMYmargin0;j++){
        
        uint idMYmargin = PartMYonLinefromLineID[j * 100 + 0];
        d3 MA = partMemYolkPos[idMYmargin] - An;
        MA /= length(MA);
        
        d2 scalPA = d2( dot(PA, normal1), dot(PA, normal2) );
        d2 scalMA = d2( dot(MA, normal1), dot(MA, normal2) );
        
        double scal = nv::dot(scalPA, scalMA) / ( length(scalPA) * length(scalMA));
        
        if(scal > scalmin){
          //insertion en par ordre de ranking
          uint k=1;
          while( PartMYrankAV[ PartMYonLinefromLineID[100*j + k] ] < PartMYrankAV[ i ] 
                        && k < PartMYonLinefromLineIDnum[j]){
            k++;
          }
          
          for(uint l= PartMYonLinefromLineIDnum[j];l>k;l--){
            PartMYonLinefromLineID[100 * j + l] = PartMYonLinefromLineID[100 * j + l - 1];
          }
          
          PartMYonLinefromLineID[100 * j + k] = i;
          PartMYonLinefromLineIDnum[j]++;
        }
      }
    }
  }
  
  //on recalcule le ranking avec la distance a lorigine da la marge
  for(uint i = 0; i < numPartMYmargin0; i++){
    uint idMYmargin = PartMYonLinefromLineID[i * 100 + 0];
    for(uint j=0;j<PartMYonLinefromLineIDnum[i];j++){
      uint idMY = PartMYonLinefromLineID[100 * i + j];
      double rank = length( partMemYolkPos[idMYmargin] - partMemYolkPos[idMY] );
      PartMYrankAV[idMY] = rank; 
    }
  }
  
  //Calcul de rankmax
  //on ne fait pas le calcul plus tot car la part vegetal (de rank rankmax) peut appartenir 
  //a plusieurs lignes
  double rankmax = .0;
  for(uint i=0;i<NUMPARTYOLKMEMBRANE;i++){
    if(state.customState.yolkMembraneActivated[i] == 0){
      if(PartMYrankAV[i] > rankmax){
        rankmax = PartMYrankAV[i];
      }
    }
  }

  //on fait des tranches latitudinales dans les memyolk
  //la premiere tranche est composée de la marge initiale
  //la derniere tranche est composée d'une seule partMY
  double layerLength = (rankmax - 2.0*.001) / (double)(NUMLATITUDE-2);
  
  uint * partMYIDfromLatitudeID = new uint[200 * NUMLATITUDE];    //200 part max par tranche
  uint * partMYIDfromLatitudeNum = new uint [NUMLATITUDE];
  memset(partMYIDfromLatitudeNum, 0, NUMLATITUDE*sizeof(uint));
  uint * partMYlatitude = new uint[NUMPARTYOLKMEMBRANE];

  for(uint i=0;i<NUMPARTYOLKMEMBRANE;i++){
    
    if(state.customState.yolkMembraneActivated[i] == 0 || state.customState.yolkMembraneEYSL[i]){
      
      double rank = PartMYrankAV[i];
      if( rank == .0 ){
        partMYlatitude[i] = 0;
      }
      else if( rank == rankmax ){
        partMYlatitude[i] = NUMLATITUDE - 1;
      }
      else{
        partMYlatitude[i] = 1 + (uint)std::floor((rank - .001) / layerLength);
      }
      partMYIDfromLatitudeID[ 200 * partMYlatitude[i] + partMYIDfromLatitudeNum[partMYlatitude[i]] ] = i;
      partMYIDfromLatitudeNum[partMYlatitude[i]]++;
    }
  }
  
  //recherche next avec tangente
  uint idmin=0;
  thrust::fill_n(state.customState.yolkMembraneNextNum.begin(), 2*NUMPARTYOLKMEMBRANE, 0);

  for(uint i=NUMLATITUDE-1;i>0;i--){

    for(uint j=0;j<partMYIDfromLatitudeNum[i];j++){
    
      double scalmin = 1000.0;
      uint idMYN = partMYIDfromLatitudeID[ 200 * i + j ];
      
      d3 N = partMemYolkPos[idMYN];
      
      uint idv1 = (uint)state.customState.yolkMembraneTangentParams[idMYN].x;
      uint idv2 = (uint)state.customState.yolkMembraneTangentParams[idMYN].z;
      
      d3 tang = state.customState.yolkMembraneTangentParams[idMYN].y * (partMemYolkPos[idv1] - N)  
                    +
                state.customState.yolkMembraneTangentParams[idMYN].w * (partMemYolkPos[idv2] - N);       
      
      double l = length(tang);
      tang /= l;
                        
      for(uint k=0;k<partMYIDfromLatitudeNum[i-1];k++){
        
        uint idMYNm1 = partMYIDfromLatitudeID[ 200 * (i-1) + k ];
        
        d3 Nm1 = partMemYolkPos[idMYNm1];
        d3 NNm1 = Nm1 - N;
        l = length(NNm1);
        NNm1 /= l;
        
        double scal = dot(NNm1, tang);
        
        double dist = length( partMemYolkPos[idMYN] - partMemYolkPos[idMYNm1] );
        
        if(scal < scalmin && dist < Ryolk * .4){
          scalmin = scal;
          idmin = idMYNm1;
        }
        
      }
      
      //si on a trouvé une part N-1 proche on ajoute
      if(scalmin != 1000.0){
        state.customState.yolkMembraneNextId[ 20 * idmin + state.customState.yolkMembraneNextNum[idmin] ] = idMYN;
        state.customState.yolkMembraneNextNum[idmin]++;

        state.customState.yolkMembraneNextId[ 20 * NUMPARTYOLKMEMBRANE + 20 * idMYN + state.customState.yolkMembraneNextNum[NUMPARTYOLKMEMBRANE+idMYN] ] = idmin;
        state.customState.yolkMembraneNextNum[NUMPARTYOLKMEMBRANE+idMYN]++;
      }
    } 
    
    //si parmi la tranche N-1, on a encore des part sans next, on cherche la part la + proche dans la tranche N
    for(uint j=0;j<partMYIDfromLatitudeNum[i-1];j++){
        
      uint idMYNm1 = partMYIDfromLatitudeID[ 200 * (i-1) + j ];
      
      if( state.customState.yolkMembraneNextNum[idMYNm1] == 0 ){
        
        double scalmin = 1000.0;
      
        d3 Nm1 = partMemYolkPos[idMYNm1];
        
        uint idv1 = (uint)state.customState.yolkMembraneTangentParams[idMYNm1].x;
        uint idv2 = (uint)state.customState.yolkMembraneTangentParams[idMYNm1].z;
      
        d3 tang = state.customState.yolkMembraneTangentParams[idMYNm1].y * (partMemYolkPos[idv1] - Nm1)  
                    +
                  state.customState.yolkMembraneTangentParams[idMYNm1].w * (partMemYolkPos[idv2] - Nm1);   
        
        tang /= length(tang);
        
        for(uint k=0;k<partMYIDfromLatitudeNum[i];k++){
        
          uint idMYN = partMYIDfromLatitudeID[ 200 * i + k ];
        
          d3 N = partMemYolkPos[idMYN];

          d3 NNm1 = Nm1 - N;
          NNm1 /= length(NNm1);
          
          double scal = dot(NNm1, tang);
          
          double dist = length(partMemYolkPos[idMYN] - partMemYolkPos[idMYNm1]);
          
          if(scal < scalmin && dist < Ryolk * .4){
            scalmin = scal;
            idmin = idMYN;
          }
        }
        
        //si on a trouvé une part N proche on ajoute
        if(scalmin != 1000.0){
          state.customState.yolkMembraneNextId[ 20 * idMYNm1 + state.customState.yolkMembraneNextNum[idMYNm1] ] = idmin;
          state.customState.yolkMembraneNextNum[idMYNm1]++;

          state.customState.yolkMembraneNextId[ 20 * NUMPARTYOLKMEMBRANE + 20 * idmin + state.customState.yolkMembraneNextNum[NUMPARTYOLKMEMBRANE+idmin] ] = idMYNm1;
          state.customState.yolkMembraneNextNum[NUMPARTYOLKMEMBRANE+idmin]++;
        }
      }   
    }


  }

  //la part du pole vegetal n a pas de voisin
  state.customState.yolkMembraneNextNum[partMYIDfromLatitudeID[200*(NUMLATITUDE-1)]] = 99999;

  // for(uint i=0; i<NUMPARTYOLKMEMBRANE; i++){
  //   printf("yolkmem %d numnext %d numprev %d \n",
  //         i, state.customState.yolkMembraneNextNum[i], 
  //         state.customState.yolkMembraneNextNum[i+NUMPARTYOLKMEMBRANE]);
  // }

  //save state as xml file
  save< State_Host >(state, "state_archive.xml");
  // load< State_Host >(state, "state_archive.xml");
 
  /**********************************/
  /**********************************/
  /**********   Param       *********/
  /**********************************/
  /**********************************/


  Param_Host param;

  param.globalDamping[0] =      3000; //666666.666667;   -> * .0045
  param.deltaTime[0] =          1.0;  //1 seconds per time step -> 6 elsewhere...

  /**** Cell Cycle *****/
  // param.cellCycleParams[0].mode = 0; //geometric progression
  // param.cellCycleParams[0].param1 = 1;
  // param.cellCycleParams[0].param2 = 1;
  // param.cellCycleParams[0].volume_ratio = .5;    
  // param.cellCycleParams[0].mPhaseLength = 10;    //10 ts equal 1 minute 
  param.cellCycleParams[0].mode = 2; //constant cycle
  param.cellCycleParams[0].param1 = 200000;
  param.cellCycleParams[0].volume_ratio = .75;   

  /***** MechaParams ****/
  for(uint i=0; i<3; i++){
    for(uint j=0; j<3; j++){
      param.mechaParams[0].maximumDistanceCoefficient[3*i+j]     = 1.2414;
      param.mechaParams[0].surfaceScaling[3*i+j]                 = 1.3697;
      param.mechaParams[0].equilibriumDistanceCoefficient[3*i+j] = 0.90469;     // 0.9523128 in 2D plane
      param.mechaParams[0].repulsionCoefficient[3*i+j]           = 4500.0; //1000000.0 * 1.0 * .0045; //500000.0; //100.0;
      param.mechaParams[0].planarRigidityCoefficient[3*i+j]      = .0; //4000.0;
    }
  }

  /*** Polarization AXes ***/
  param.numPolarizationAxes[0] = 2;

  // posterior border axis, following Delta
  param.polarizationAxisParams[0].idlig                   = 0; // Evl-lig
  param.polarizationAxisParams[0].compMode                = 0; // weighted neighboring link
  param.polarizationAxisParams[0].apicoBasalInEpithelium  = 0; 
  
  // anterior border axis, following Delto
  param.polarizationAxisParams[1].idlig                   = 1; // Yolk-lig
  param.polarizationAxisParams[1].compMode                = 0; // weighted neighboring link
  param.polarizationAxisParams[1].apicoBasalInEpithelium  = 0;

  /*** Ligands ***/
  param.numLigands[0] = 2;

  param.ligandParams[0].diffusion = 10.0;
  param.ligandParams[0].kappa =  0.02; 
  sprintf(param.ligandParams[0].name, "Evl-lig");

  param.ligandParams[1].diffusion = 10.0;
  param.ligandParams[1].kappa =  0.02; 
  sprintf(param.ligandParams[1].name, "Yolk-lig");

  /**** Proteins *******/
  param.numProteins[0] = 1;

  param.proteins[0].kappa = .015;
  sprintf(param.proteins[0].name, "Ubiquitous");

  /*** ProteinNodes ****/
  param.numProteinNodes[0] = 1;

  // Ubiquitous
  param.proteinNodes[0].outputProteinID = 0;
  param.proteinNodes[0].Xmin = d3(-10000.0,-10000.0,-10000.0);
  param.proteinNodes[0].Xmax = d3(10000.0,10000.0,10000.0);
  param.proteinNodes[0].tmin = 0;
  param.proteinNodes[0].tmax = 10000;
  param.proteinNodes[0].quantity = 1.0;

  /*** PPInteractions ****/
  param.numPPInteractions[0] = 0;
  
  /*** Receptors ****/
  param.numReceptors[0] = 0;

  /*** TransReceptors ****/
  param.numTransReceptors[0] = 0;

  /*** Secretors ***/
  param.numSecretors[0] = 0;

  /*** Genes ****/
  param.numGenes[0] = 0;

  /*** Polarization Nodes ***/
  param.numPolarizationNodes[0] = 1;

  param.polarizationNodes[0].axisID = 0;                      // Lig polarization axis: 0 Evl-lig, 1 Yolk-lig
  param.polarizationNodes[0].regEl.logicalFunction = 0;       // AND function
  param.polarizationNodes[0].regEl.numInputProtein = 1;   
  param.polarizationNodes[0].regEl.inputProteinID[0] = 0;     // Ubiquitous
  param.polarizationNodes[0].regEl.inputThreshold[0] = 5.0;
  param.polarizationNodes[0].regEl.inputType[0]      = 1;     

  /*** EpiPolarization Nodes ***/
  param.numEpiPolarizationNodes[0] = 0;
  
  /*** Adhesion Nodes ***/
  param.numAdhesionNodes[0] = 1;
  param.adhesionNodes[0].mode = 3;        // lazy mode: constant value, independent from protein concentration
  param.adhesionNodes[0].k_adh = 4500.0; //1000000.0 * 1.0* .0045; //500000.0; //100.0;
  param.adhesionNodes[0].proteinID = 0;   // unused in lazy mode

  /*** Celltype Nodes ***/
  param.cellTypeNodes[0].numInputProtein = 1;       // Mesenchymal cells
  param.cellTypeNodes[0].logicalFunction = 0;       // AND
  param.cellTypeNodes[0].inputProteinID[0] = 0;     // Ubiquitous
  param.cellTypeNodes[0].inputThreshold[0] = 50.0;
  param.cellTypeNodes[0].inputType[0]      = 1;

  param.cellTypeNodes[1].numInputProtein = 0;       // Epithelial cells

  /*** Protrusion Node ***/
  param.protrusionNode[0].regEl.logicalFunction   = 0;       // AND function
  param.protrusionNode[0].regEl.numInputProtein   = 1;   
  param.protrusionNode[0].regEl.inputProteinID[0] = 0;     // Ubiquitous
  param.protrusionNode[0].regEl.inputThreshold[0] = 5.0;
  param.protrusionNode[0].regEl.inputType[0]      = 1;     
  param.protrusionNode[0].force                   = .002; //3560; //1; //3560;
  param.protrusionNode[0].adhesionID              = 0;     // todo change to real adhesion mol

  /*** Bipolarity Node ***/
  param.bipolarityNode[0].regEl.logicalFunction   = 0;       // AND function
  param.bipolarityNode[0].regEl.numInputProtein   = 1;   
  param.bipolarityNode[0].regEl.inputProteinID[0] = 0;     // Ubiquitous
  param.bipolarityNode[0].regEl.inputThreshold[0] = 5.0;
  param.bipolarityNode[0].regEl.inputType[0]      = 1;  

  /***** Display GRN specifications ****/
  param.display();

  CustomParams cp;

  cp.yolkInteriorCmax = 1.2414;
  cp.yolkInteriorSurfaceScaling = 1.3697;
  cp.yolkInteriorAttractionCoefficient = 2500.0; //1025.0;
  cp.yolkInteriorRepulsionCoefficient = 2500.0; //2325.0;// 100.0;

  cp.yolkMembraneStiffness = 5.5;
  cp.yolkMembraneRLCoeff = .95; //.95;
  
  cp.cellsYolkCmax = 1.2414;
  cp.cellsYolkSurfaceScaling = 1.3697;
  cp.cellsYolkEquilibriumDistance = 0.90469;
  cp.cellsYolkAttractionCoefficient = 35000.0; //35000.0;     //10.0; higher than OLD
  cp.cellsYolkRepulsionCoefficient = 10000.0;
  cp.cellsEvlCmax = 1.2414;
  cp.cellsEvlSurfaceScaling = 1.3697;
  cp.cellsEvlEquilibriumDistance = 0.90469;
  cp.cellsEvlAttractionCoefficient = 500.0; //2250.0;
  cp.cellsEvlRepulsionCoefficient = 4500.0; //6750.0;

  cp.yolkMarginEvlStiffness = 10.0; //5.0;//10.0;

  double apf_hc = PI / (3*sqrtf(2));
  cp.evlRLCoeff = pow(apf_hc, 1.0/3.0);

  cp.evlCmax = 3.0; 
  cp.evlStiffness = 18.0;

  cp.evlRadiusAB = Ryolk * 0.038836;

  cp.marginResistance = .005; //.017; //5.33; smaller than OLD * .01

  cp.evlGrowthThreshold     = 8.0; //56.67 * .0045;
  cp.evlLateralGrowthRatio  = 1.01;
  cp.evlRadiusLimit         = 0.077672; //2 * Ryolk*0.038836;

  cp.evlLigandId        = 0;
  cp.evlLigandUpdate    = 100.0;
  cp.yolkLigandId       = 1;
  cp.yolkLigandUpdate   = 100.0;

  param.customParam[0] = cp;

  save< Param_Host >(param, "param_archive.xml");
}
