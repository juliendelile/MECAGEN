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

#include "customdisplay.hpp"
#include "glwidget.hpp"
#include <stdio.h>

namespace mg{

  void printcustom(EmbryoState *mg_buf, GLWidget * glw){

    // Draw yolk membrane particles
    // for(uint i=0; i < NUMPARTYOLKMEMBRANE; i++ ){ 
    //   if(glw->slice(mg_buf->customStateBuffer.yolkPosition[i])){
    //     if(mg_buf->customStateBuffer.yolkMembraneActivated[i] == 1){
    //       if(mg_buf->customStateBuffer.yolkMembraneEYSL[i] == 1){
    //         glColor3f(.0,.80,.10);
    //       }
    //       else{
    //         glColor3f(.75,.75,.0);
    //       }
    //       glPushMatrix();
    //       glTranslated(
    //         mg_buf->customStateBuffer.yolkPosition[i].x,
    //         mg_buf->customStateBuffer.yolkPosition[i].y,
    //         mg_buf->customStateBuffer.yolkPosition[i].z);
    //       glutSolidSphere(.01,10,10);
    //       // glutSolidSphere(APF_2D_RAD*mg_buf->customStateBuffer.yolkMembraneRadius[0],10,10);

    //       glPopMatrix();
    //     }
    //   }
    // }

    // Draw yolk interior particles
    for(uint i=NUMPARTYOLKMEMBRANE; i < NUMPARTYOLK; i++ ){ 
      if(glw->slice(mg_buf->customStateBuffer.yolkPosition[i])){
        // glColor3f(1.0,.0,.0);
        // glColor3f(.7,.7,.7);
        glColor3f(.75,.75,.0);
        glPushMatrix();
        glTranslated(
          mg_buf->customStateBuffer.yolkPosition[i].x,
          mg_buf->customStateBuffer.yolkPosition[i].y,
          mg_buf->customStateBuffer.yolkPosition[i].z);
        
        // glutSolidSphere(.01,10,10); 

        GLUquadricObj *quadObj = gluNewQuadric();
        gluQuadricDrawStyle(quadObj, GLU_FILL);
        gluQuadricNormals(quadObj, GLU_SMOOTH);
        gluSphere(quadObj, .01, 10, 10);
        gluDeleteQuadric(quadObj);


        // glutSolidSphere(APF_2D_RAD*mg_buf->customStateBuffer.yolkInteriorRadius[0],10,10); 
        glPopMatrix();
      }
    }

    // Draw neighbor vectors between yolk membrane particules
    glBegin(GL_LINES);  
    for (uint i = 0; i < NUMPARTYOLKMEMBRANE; i++){ 
      if(glw->slice(mg_buf->customStateBuffer.yolkPosition[i])){
        // glColor3f(1.0, .5, .0);
        // glColor3f(.7,.7,.7);
        glColor3f(.75,.75,.0);
        for (uint k = 0; k < mg_buf->customStateBuffer.yolkMembraneNeighbNum[i+NUMPARTYOLKMEMBRANE]; k++){ 
          uint ineigh = NUMNEIGHMYMYmax*i + k;
          uint idneigh = mg_buf->customStateBuffer.yolkMembraneNeighbId[ ineigh ];
          // if( k < mg_buf->customStateBuffer.yolkMembraneNeighbNum[i] ){  
          //   glColor3f(1.0, .5, .0);
          // }
          // else{ 
          //   glColor3f(1.0, .1, .0);
          // }
          glVertex3dv((double*)&(mg_buf->customStateBuffer.yolkPosition[i]));
          glVertex3dv((double*)&(mg_buf->customStateBuffer.yolkPosition[idneigh]));
        }
      }
    }
    glEnd();

    // Yolk Interior Neighborhood
    glBegin(GL_LINES);  
    for (uint i = 0; i < NUMPARTYOLKINTERIOR; i++){ 
      if(glw->slice(mg_buf->customStateBuffer.yolkPosition[i+NUMPARTYOLKMEMBRANE])){
        // glColor3f(.90, .90, .50);
        // glColor3f(.7,.7,.7);
        glColor3f(.75,.75,.0);
        for (uint k = 0; k < mg_buf->customStateBuffer.yolkInteriorTopologicalNeighbNum[i]; k++){ 
          uint ineigh = NUMNEIGHBTOPOmax*i + k;
          uint idneigh = mg_buf->customStateBuffer.yolkInteriorTopologicalNeighbId[ ineigh ];
          glVertex3dv((double*)&(mg_buf->customStateBuffer.yolkPosition[i+NUMPARTYOLKMEMBRANE]));
          glVertex3dv((double*)&(mg_buf->customStateBuffer.yolkPosition[idneigh]));
        }
      }
    }
    glEnd();

    // EVL centers
    for(uint i=0; i < mg_buf->customStateBuffer.numPartEVL[0]; i++ ){ 
      if(glw->slice(mg_buf->customStateBuffer.evlPosition[i])){
        glColor3f(0.40625,.0, 0.64453125);
        glPushMatrix();
        glTranslated(
          mg_buf->customStateBuffer.evlPosition[i].x,
          mg_buf->customStateBuffer.evlPosition[i].y,
          mg_buf->customStateBuffer.evlPosition[i].z);

        // glutSolidSphere(.01,10,10); 
        
        GLUquadricObj *quadObj = gluNewQuadric();
        gluQuadricDrawStyle(quadObj, GLU_FILL);
        gluQuadricNormals(quadObj, GLU_SMOOTH);
        gluSphere(quadObj, .01, 10, 10);
        gluDeleteQuadric(quadObj);

        glPopMatrix();
      }
    }

    // EVL Normales
    // glBegin(GL_LINES);  
    // for (uint i = 0; i < mg_buf->customStateBuffer.numPartEVL[0]; i++){ 
    //   if(glw->slice(mg_buf->customStateBuffer.evlPosition[i])){
    //     glColor3f(.0, 1.0, 1.0);
    //     glVertex3dv((double*)&(mg_buf->customStateBuffer.evlPosition[i]));
    //     d3 normal = mg_buf->customStateBuffer.evlPosition[i] + .1 * mg_buf->customStateBuffer.evlNormal[i];
    //     glVertex3dv((double*)&normal);
    //   }
    // }
    // glEnd();
    
    // EVL-EVL Neighborhood
    glBegin(GL_LINES);  
    for (uint i = 0; i < mg_buf->customStateBuffer.numPartEVL[0]; i++){ 
      if(glw->slice(mg_buf->customStateBuffer.evlPosition[i])){
        // glColor3f(1.0, 1.0, .0);
        glColor3f(.7,.7,.7);
        for (uint k = 0; k < mg_buf->customStateBuffer.evlTopologicalNeighbNum[i]; k++){ 
          uint ineigh = NUMNEIGHBTOPOmax*i + k;
          uint idneigh = mg_buf->customStateBuffer.evlTopologicalNeighbId[ ineigh ];
          glVertex3dv((double*)&(mg_buf->customStateBuffer.evlPosition[i]));
          glVertex3dv((double*)&(mg_buf->customStateBuffer.evlPosition[idneigh]));
        }
      }
    }
    glEnd();
    
    // Yolk membrane - IC Neighborhood
    glBegin(GL_LINES);  
    for (uint i = 0; i < NUMPARTYOLKMEMBRANE; i++){ 
      if(glw->slice(mg_buf->customStateBuffer.yolkPosition[i])){
        // glColor3f(1.0, .5, .5);
        glColor3f(.7,.7,.7);
        for (uint k = 0; k < mg_buf->customStateBuffer.yolkCellsNeighbNum[i]; k++){ 
          uint idneigh = mg_buf->customStateBuffer.yolkCellsNeighbId[i];
          glVertex3dv((double*)&(mg_buf->customStateBuffer.yolkPosition[i]));
          glVertex3dv((double*)&(mg_buf->cellPosition[idneigh]));
        }
      }
    }
    glEnd();
    
    // EVL - IC Neighborhood
    glBegin(GL_LINES);  
    for (uint i = 0; i < mg_buf->numCells; i++){ 
      if(glw->slice(mg_buf->cellPosition[i])){
        // glColor3f(.5, .5, 1.0);
        glColor3f(.7,.7,.7);
        for (uint k = 0; k < mg_buf->customStateBuffer.cellsEvlNeighbNum[i]; k++){ 
          uint idneigh = mg_buf->customStateBuffer.cellsEvlNeighbId[i];
          glVertex3dv((double*)&(mg_buf->cellPosition[i]));
          glVertex3dv((double*)&(mg_buf->customStateBuffer.evlPosition[idneigh]));
        }
      }
    }
    glEnd();
    
    // Yolk Margin - EVL Neighborhood
    glBegin(GL_LINES);
    for (uint i = 0; i < NUMPARTYOLKMEMBRANE; i++){ 
      if(glw->slice(mg_buf->customStateBuffer.yolkPosition[i])){
        // glColor3f(.5, 1.0, .5);
        glColor3f(.7,.7,.7);
        for (uint k = 0; k < mg_buf->customStateBuffer.yolkMarginEvlTopologicalNeighbNum[i]; k++){ 
          uint ineigh = NUMNEIGHBTOPOmax*i + k;
          uint idneigh = mg_buf->customStateBuffer.yolkMarginEvlTopologicalNeighbId[ineigh];
          glVertex3dv((double*)&(mg_buf->customStateBuffer.yolkPosition[i]));
          glVertex3dv((double*)&(mg_buf->customStateBuffer.evlPosition[idneigh]));
        }
      }
    }
    glEnd();
   
    // Draw yolk membrane particules' tangent vector. 
    // glBegin(GL_LINES);
    // glLineWidth(4);
    // for (uint i = 0; i < NUMPARTYOLKMEMBRANE; i++){ 
    //   d3 A = mg_buf->customStateBuffer.yolkPosition[i];
    //   if(glw->slice(mg_buf->customStateBuffer.yolkPosition[i])){
    //     glColor3f(.0, 1.0, .0);
        
    //     d3 AB = mg_buf->customStateBuffer.yolkPosition[ (uint)mg_buf->customStateBuffer.yolkMembraneTangentParams[i].x ] - A;
    //     d3 AC = mg_buf->customStateBuffer.yolkPosition[ (uint)mg_buf->customStateBuffer.yolkMembraneTangentParams[i].z ] - A;
    //     d3 ptangent = mg_buf->customStateBuffer.yolkMembraneTangentParams[i].y * AB
    //                 + mg_buf->customStateBuffer.yolkMembraneTangentParams[i].w * AC;
    //     ptangent/=length(ptangent);

    //     glVertex3dv((double*)&(mg_buf->customStateBuffer.yolkPosition[i]));
    //     d3 tip = A + .03 * ptangent;
    //     glVertex3d(tip.x, tip.y, tip.z);
    //   }
    // }
    // glLineWidth(1);
    // glEnd();

    // Draw yolk membrane particules' normal vector
    // glBegin(GL_LINES);
    // glLineWidth(4);
    // for (uint i = 0; i < NUMPARTYOLKMEMBRANE; i++)  
    // {
    //   if(glw->slice(mg_buf->customStateBuffer.yolkPosition[i])){
    //     d3 normal = d3(.0);

    //     d3 A = mg_buf->customStateBuffer.yolkPosition[i];

    //     for(uint j=0;j<mg_buf->customStateBuffer.yolkMembraneNeighbNum[i] - 1;j++){
          
    //       uint idneigh = mg_buf->customStateBuffer.yolkMembraneNeighbId[NUMNEIGHMYMYmax*i + j];
    //       d3 B = mg_buf->customStateBuffer.yolkPosition[idneigh];
    //       idneigh = mg_buf->customStateBuffer.yolkMembraneNeighbId[NUMNEIGHMYMYmax*i + j + 1];
    //       d3 C = mg_buf->customStateBuffer.yolkPosition[idneigh];
          
    //       d3 normaltri = cross(B-A,C-A);
    //       normaltri /= length(normaltri);
          
    //       normal += normaltri;
    //     }
        
    //     uint idneigh = mg_buf->customStateBuffer.yolkMembraneNeighbId[NUMNEIGHMYMYmax*i + mg_buf->customStateBuffer.yolkMembraneNeighbNum[i] - 1];
    //     d3 B = mg_buf->customStateBuffer.yolkPosition[idneigh];
    //     idneigh = mg_buf->customStateBuffer.yolkMembraneNeighbId[NUMNEIGHMYMYmax*i + 0];
    //     d3 C = mg_buf->customStateBuffer.yolkPosition[idneigh];
        
    //     d3 normaltri = cross(B-A,C-A);
    //     normaltri /= length(normaltri);
        
    //     normal += normaltri;
        
    //     normal /= length(normal);

    //     glColor3f(.0, 1.0, 1.0);
     
    //     glVertex3dv((double*)&(mg_buf->customStateBuffer.yolkPosition[i]));
    //     d3 tip = A + .03 * normal;
    //     glVertex3d(tip.x, tip.y, tip.z);
    //   }
    // }
    // glLineWidth(1);
    // glEnd();

    //Yolk River
    // glLineWidth(50);    
    // glBegin(GL_LINES); 
    // uint numnei;
    // for (uint i = 0; i < NUMPARTYOLKMEMBRANE; i++)  { 
    //   if(glw->slice(mg_buf->customStateBuffer.yolkPosition[i])){
    //     if(mg_buf->customStateBuffer.yolkMembraneActivated[i] == 0 || mg_buf->customStateBuffer.yolkMembraneEYSL[i]){
          
    //       d3 A = mg_buf->customStateBuffer.yolkPosition[i];
    //       numnei = mg_buf->customStateBuffer.yolkMembraneNextNum[i];
    //       if( numnei != 99999 ){        

    //         for(uint j=0;j<numnei;j++){
    //           glColor3f(1.0, .0, .0);
    //           glVertex3d( A.x, A.y, A.z );  
    //           uint idneigh = mg_buf->customStateBuffer.yolkMembraneNextId[ 20 * i + j ];
    //           d3 B = mg_buf->customStateBuffer.yolkPosition[idneigh];
    //           glColor3f(.0, 1.0, .0);
    //           glVertex3d( B.x,B.y,B.z); 
    //        }
    //       }
    //     }
    //   }
    // }
    // glEnd();
    // glLineWidth(1);

  }


  void printcustom2D(EmbryoState *mg_buf, float *modelview, float *projection, float w, float h, GLWidget * glw){

    for(uint i=0; i < NUMPARTYOLKMEMBRANE; i++ ){ 
      d3 pos = mg_buf->customStateBuffer.yolkPosition[i];

      if(glw->slice(pos)){

        d4 a(
                modelview[0]*pos.x + modelview[4]*pos.y + modelview[8]*pos.z + modelview[12],
                modelview[1]*pos.x + modelview[5]*pos.y + modelview[9]*pos.z + modelview[13],
                modelview[2]*pos.x + modelview[6]*pos.y + modelview[10]*pos.z + modelview[14],
                modelview[3]*pos.x + modelview[7]*pos.y + modelview[11]*pos.z + modelview[15]
            );
        d4 o(
                projection[0]*a.x + projection[4]*a.y + projection[8]*a.z + projection[12]*a.w,
                projection[1]*a.x + projection[5]*a.y + projection[9]*a.z + projection[13]*a.w,
                projection[2]*a.x + projection[6]*a.y + projection[10]*a.z + projection[14]*a.w,
                projection[3]*a.x + projection[7]*a.y + projection[11]*a.z + projection[15]*a.w
            );

        if(o.z > -1){

          if (o.w != 0.0){
            o.x /= o.w;
            o.y /= o.w;
          }

          double px = w * (1.0 + o.x) / 2.0;
          double py = h * (1.0 + o.y) / 2.0;
          char c[256];
          glColor3f (.0,.0,.0);
          glRasterPos2f(px,py);

          sprintf(c,"%d ",i);

          int id=0;
          while(c[id] != '\0'){
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15,c[id]);
            id++;
          }
        }
      }
    }


    for(uint i=0; i < mg_buf->customStateBuffer.numPartEVL[0]; i++ ){ 
      
      d3 pos = mg_buf->customStateBuffer.evlPosition[i];
      
      if(glw->slice(pos)){

        d4 a(
                modelview[0]*pos.x + modelview[4]*pos.y + modelview[8]*pos.z + modelview[12],
                modelview[1]*pos.x + modelview[5]*pos.y + modelview[9]*pos.z + modelview[13],
                modelview[2]*pos.x + modelview[6]*pos.y + modelview[10]*pos.z + modelview[14],
                modelview[3]*pos.x + modelview[7]*pos.y + modelview[11]*pos.z + modelview[15]
            );
        d4 o(
                projection[0]*a.x + projection[4]*a.y + projection[8]*a.z + projection[12]*a.w,
                projection[1]*a.x + projection[5]*a.y + projection[9]*a.z + projection[13]*a.w,
                projection[2]*a.x + projection[6]*a.y + projection[10]*a.z + projection[14]*a.w,
                projection[3]*a.x + projection[7]*a.y + projection[11]*a.z + projection[15]*a.w
            );

        if(o.z > -1){

          if (o.w != 0.0){
            o.x /= o.w;
            o.y /= o.w;
          }

          double px = w * (1.0 + o.x) / 2.0;
          double py = h * (1.0 + o.y) / 2.0;
          char c[256];
          glColor3f (1.0,1.0,.0);
          glRasterPos2f(px,py);

          sprintf(c,"%d ",i);

          int id=0;
          while(c[id] != '\0'){
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15,c[id]);
            id++;
          }
        }
      }
    }
  }
}
