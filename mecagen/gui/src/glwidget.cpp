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

#include "glwidget.hpp"
#include "customdisplay.hpp"

#ifdef CUDA_VBO
#include "cuda/vbo.h"
#include <cuda.h>
#include <cuda_gl_interop.h>
#endif

#include <stdlib.h>

namespace mg{

  GLfloat projection[16];
  GLfloat modelview[16];
  double mouseX;
  double mouseY;

  //
  // Constructor/destructor:
  //

  GLWidget::GLWidget(MetaParam<HOST> * _mp, Param<HOST> * _p, QGLFormat format, QWidget *parent) :
    QGLWidget(QGLFormat(QGL::SampleBuffers), parent),
    metaParam(_mp),
    param(_p),
    mg_buf(NULL),
    recording(false),
    orthoProjection(false),
    twoDinfo(false),
    snapshot_counter(0),
    record_single_snapshot(false)

    #ifdef CUDA_VBO
      ,
      newParamAvailable(0),
      newDataAvailable(0)
    #endif
  {
    // Camera settings:
    xRot  = yRot  = zRot  = 0;
    cam_x = cam_y = cam_z = 0.0;
    eyeZ  = -55.0;
    scale = 35.0; //.40;//0.25;

    centralPoint = d3( .0 );

    // The urshin's four colors, on by cell population
    sea_urchin_pop_colors[0] = mg::f3(.3608,.0,.3020);
    sea_urchin_pop_colors[1] = mg::f3(1.0,.4745,1.0);
    sea_urchin_pop_colors[2] = mg::f3(.8941,.0745,.0);
    sea_urchin_pop_colors[3] = mg::f3(.0510,.7784,.8196);

    connect(&timer, SIGNAL(timeout()), this, SLOT(updateGL()));
    timer.start(50);

    numCells = 0;

    //slicers
    SlicerXmax = 1000.0;
    SlicerYmax = 1000.0;;
    SlicerZmax = 1000.0;
    SlicerXmin = -1000.0;
    SlicerYmin = -1000.0;
    SlicerZmin = -.01 * metaParam->displayScale[0]; //-1000.0;

    //Embryo axes
    // LRaxis = d3(1.0,.0,.0);
    // APaxis = d3(.0,1.0,.0);
    // AVaxis = d3(.0,.0,1.0);
    LRaxis = metaParam->embryoAxes[0];
    APaxis = metaParam->embryoAxes[1];
    AVaxis = metaParam->embryoAxes[2];

    //GRN display
    for(uint i=0; i<NUMPROTEINmax; i++){
      displayParams.proteinColorR[i] = (float)rand()/(float)RAND_MAX; 
      displayParams.proteinColorG[i] = (float)rand()/(float)RAND_MAX; 
      displayParams.proteinColorB[i] = (float)rand()/(float)RAND_MAX; 
      displayParams.proteinThreshold[i] = 100.0; 
    }

    for(uint i=0; i<NUMLIGmax; i++){
      displayParams.ligandColorR[i] = (float)rand()/(float)RAND_MAX; 
      displayParams.ligandColorG[i] = (float)rand()/(float)RAND_MAX; 
      displayParams.ligandColorB[i] = (float)rand()/(float)RAND_MAX; 
      displayParams.ligandThreshold[i] = 100.0; 
    }

    // Signal-mediated toggle switch
    // displayParams.proteinColorR[0] = 75/255.0;displayParams.proteinColorG[0] = 68/255.0; displayParams.proteinColorB[0] = 135/255.0;  // X
    // displayParams.proteinColorR[1] = 255/255.0;displayParams.proteinColorG[1] = 170/255.0; displayParams.proteinColorB[1] = 170/255.0;  // Frizzled
    // displayParams.proteinColorR[2] = 0/255.0;displayParams.proteinColorG[2] = 255/255.0; displayParams.proteinColorB[2] = 0/255.0;  // XIAP-inducer
    // displayParams.proteinColorR[3] = 0/255.0;displayParams.proteinColorG[3] = 163/255.0; displayParams.proteinColorB[3] = 223/255.0;  // Beta-cat
    // displayParams.proteinColorR[4] = 223/255.0;displayParams.proteinColorG[4] = 216/255.0; displayParams.proteinColorB[4] = 0/255.0; displayParams.proteinThreshold[4] = 50.0; // XIAP
    // displayParams.proteinColorR[5] = 0/255.0;displayParams.proteinColorG[5] = 128/255.0; displayParams.proteinColorB[5] = 0/255.0;  // Tcf
    // displayParams.proteinColorR[6] = 255/255.0;displayParams.proteinColorG[6] = 102/255.0; displayParams.proteinColorB[6] = 0/255.0;  // Gro
    // displayParams.proteinColorR[7] = 181/255.0;displayParams.proteinColorG[7] = 0/255.0; displayParams.proteinColorB[7] = 195/255.0; displayParams.proteinThreshold[7] = 60.0; // Tcf +
    // displayParams.proteinColorR[8] = 55/255.0;displayParams.proteinColorG[8] = 255/255.0; displayParams.proteinColorB[8] = 50/255.0;  // Tcf -
    // displayParams.proteinColorR[9] = 0/255.0;displayParams.proteinColorG[9] = 0/255.0; displayParams.proteinColorB[9] = 255/255.0;  // Target
    // displayParams.proteinColorR[10] = 78/255.0;displayParams.proteinColorG[10] = 90/255.0; displayParams.proteinColorB[10] = 177/255.0; displayParams.proteinThreshold[10] = 40.0; // Gro^Ubi
    // displayParams.ligandColorR[0] = 253/255.0;displayParams.ligandColorG[0] = 0/255.0; displayParams.ligandColorB[0] = 0/255.0; displayParams.ligandThreshold[0] = 50.0;  // Wnt
    
    //Boundary Formation
    // displayParams.proteinColorR[0] = 76/255.0;displayParams.proteinColorG[0] = 70/255.0; displayParams.proteinColorB[0] = 135/255.0;  // X
    // displayParams.proteinColorR[1] = 1/255.0;displayParams.proteinColorG[1] = 1/255.0; displayParams.proteinColorB[1] = 255/255.0;  // Delta
    // displayParams.proteinColorR[2] = 217/255.0;displayParams.proteinColorG[2] = 163/255.0; displayParams.proteinColorB[2] = 242/255.0;  // Delto
    // displayParams.proteinColorR[3] = 255/255.0;displayParams.proteinColorG[3] = 170/255.0; displayParams.proteinColorB[3] = 170/255.0;  // Notch
    // displayParams.proteinColorR[4] = 0/255.0;displayParams.proteinColorG[4] = 163/255.0; displayParams.proteinColorB[4] = 222/255.0;  // Epi-inducer
    // displayParams.proteinColorR[5] = 110/255.0;displayParams.proteinColorG[5] = 227/255.0; displayParams.proteinColorB[5] = 28/255.0;  // Epi
    // displayParams.proteinColorR[7] = 232/255.0;displayParams.proteinColorG[7] = 175/255.0; displayParams.proteinColorB[7] = 83/255.0;  // Epi2-inducer
    // displayParams.proteinColorR[8] = 255/255.0;displayParams.proteinColorG[8] = 66/255.0; displayParams.proteinColorB[8] = 0/255.0;  // Epi2
    // displayParams.proteinColorR[10] = 255/255.0;displayParams.proteinColorG[10] = 0/255.0; displayParams.proteinColorB[10] = 166/255.0;  // Anterior
    // displayParams.ligandColorR[0] = 1/255.0;displayParams.ligandColorG[0] = 1/255.0; displayParams.ligandColorB[0] = 255/255.0;  // Delta-lig
    // displayParams.ligandColorR[1] = 217/255.0;displayParams.ligandColorG[1] = 163/255.0; displayParams.ligandColorB[1] = 242/255.0;  // Delto-lig
    
    displayParams.noneColorR=.2;
    displayParams.noneColorG=.2;
    displayParams.noneColorB=.2; 

    displayParams.colorByType = 0;
    displayParams.typeColorR[0] = 184.0/255.0;
    displayParams.typeColorG[0] = 184.0/255.0;
    displayParams.typeColorB[0] = 184.0/255.0;
    displayParams.typeColorR[1] = 93.0/255.0;
    displayParams.typeColorG[1] = 211.0/255.0;
    displayParams.typeColorB[1] = 212.0/255.0;
    displayParams.typeColorR[2] = 243.0/255.0;
    displayParams.typeColorG[2] = 210.0/255.0;
    displayParams.typeColorB[2] = 154.0/255.0;
    displayParams.typeColorR[3] = 33.0/255.0;
    displayParams.typeColorG[3] = 167.0/255.0;
    displayParams.typeColorB[3] = 72.0/255.0;

    displayParams.drawAxes = 0;
    displayParams.candAxesId = -1;
    displayParams.axeAB = -1;
    displayParams.axe1 = -1;
  }

  GLWidget::~GLWidget() { }

  //
  // Hints:
  //

  QSize GLWidget::minimumSizeHint() const { return QSize(50, 50); }

  QSize GLWidget::sizeHint() const { return QSize(1000, 1000); }
  // QSize GLWidget::sizeHint() const { return QSize(700, 500); }

  //
  // Rotation management: TODO: do this better http://qt-project.org/doc/qt-5.0/qtopengl/cube.html
  //

  static void qNormalizeAngle(int &angle)
  {
    while (angle < 0){ angle += 360 * 16; }
    while (angle > 360 * 16){ angle -= 360 * 16; }
  }

  void GLWidget::setXRotation(int angle)
  {
    qNormalizeAngle(angle);
    if (angle != xRot) {
      xRot = angle;
      emit xRotationChanged(angle);
      updateGL();
    }
  }

  void GLWidget::setYRotation(int angle)
  {
    qNormalizeAngle(angle);
    if (angle != yRot) {
      yRot = angle;
      emit yRotationChanged(angle);
      updateGL();
    }
  }

  void GLWidget::setZRotation(int angle)
  {
    qNormalizeAngle(angle);
    if (angle != zRot) {
      zRot = angle;
      emit zRotationChanged(angle);
      updateGL();
    }
  }

  void GLWidget::setSlicerXmax(int i)
  {
    SlicerXmax = metaParam->displayScale[0] * (double)i/100.0;
  }

  void GLWidget::setSlicerXmin(int i)
  {
    SlicerXmin = metaParam->displayScale[0] * (double)i/100.0;
  }

  void GLWidget::setSlicerYmax(int i)
  {
    SlicerYmax = metaParam->displayScale[0] * (double)i/100.0;
  }

  void GLWidget::setSlicerYmin(int i)
  {
    SlicerYmin = metaParam->displayScale[0] * (double)i/100.0;
  }

  void GLWidget::setSlicerZmax(int i)
  {
    SlicerZmax = metaParam->displayScale[0] * (double)i/100.0;
  }

  void GLWidget::setSlicerZmin(int i)
  {
    SlicerZmin = metaParam->displayScale[0] * (double)i/100.0;
  }

  void GLWidget::setOrthoProj()
  {
    orthoProjection = true;
  }

  void GLWidget::setPerspProj()
  {
    orthoProjection = false;
  }

  void GLWidget::set2Dinfo(bool on)
  {
    if(on){
      twoDinfo = true;
    }
    else{
      twoDinfo = false;
    }
  }
  
  void GLWidget::setColByType(bool on)
  {
    if(on){
      displayParams.colorByType = 1;
    }
    else{
      displayParams.colorByType = 0;
    }
    #ifdef CUDA_VBO 
      newDisplayParamVBO();
    #endif
  }

  void GLWidget::newDisplayParamVBO(){ 
    #ifdef CUDA_VBO 
      newParamAvailable = 1;
    #endif
  }

  void GLWidget::setDrawAxes(bool on){
    if(on){
      displayParams.drawAxes = 1;
    }
    else{
      displayParams.drawAxes = 0;
    }
    #ifdef CUDA_VBO 
      newDisplayParamVBO();
    #endif
  }

  void GLWidget::setAxesId(int i){
    std::cout << "Axes id " << i << std::endl;

    if(i < param->numPolarizationAxes[0]){
      displayParams.candAxesId = i;
      displayParams.axeAB = -1;
      displayParams.axe1 = -1;
    }
    else if(i == param->numPolarizationAxes[0]){
      displayParams.candAxesId = -1;
      displayParams.axeAB = 1;
      displayParams.axe1 = -1;
    }
    else if(i == param->numPolarizationAxes[0] + 1){
      displayParams.candAxesId = -1;
      displayParams.axeAB = -1;
      displayParams.axe1 = 1;
    }

    #ifdef CUDA_VBO 
      newDisplayParamVBO();
    #endif
  }

  //
  // Mouse management:
  //

  void GLWidget::mousePressEvent(QMouseEvent *event) { 

    lastPos = event->pos(); 

    if (event->buttons() & Qt::RightButton){
      mouseX = event->pos().x();
      mouseY = height()-event->pos().y();
      printf("pick a center glwin\n");  
      pickaCenter();
    }

  }

  void GLWidget::mouseMoveEvent(QMouseEvent *event)
  {
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    if (event->buttons() & Qt::LeftButton) {
      setXRotation(xRot + 8 * dy);
      setYRotation(yRot + 8 * dx);
    } else if (event->buttons() & Qt::RightButton) {
      setXRotation(xRot + 8 * dy);
      setZRotation(zRot + 8 * dx);
    }
    lastPos = event->pos();
  }

  //
  // Timer management
  //

  void GLWidget::timerEvent(QTimerEvent *e){
    // Update scene
    updateGL();
  }

  //
  // OpenGL management:
  //

  void GLWidget::initializeGL() {

    // printf("initgl0\n");
    // funcCudaGetLastError();
    // printf("initgl0\n");

    GLenum err = glewInit();
    if (GLEW_OK != err) { printf("Error: %s\n", glewGetErrorString(err)); }

    qglClearColor(Qt::white);
    // qglClearColor(QColor(100,100,100));

   

#ifdef CUDA_VBO
    initShaders();
    initData();
#endif
    // Glut initialization:
    char text[] = {"mg"};
    char *argv[] = { text, NULL};
    int argc = sizeof(argv) / sizeof(char*) - 1;
    glutInit(&argc, argv);

    // Use QBasicTimer because its faster than QTimer
    // timer.start(50, this);
  }


  void GLWidget::paintGL()
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    double w = width();
    double h = height();
    double aspect = w/h;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    if(orthoProjection){
      glOrtho(-aspect, aspect, -1, 1, 2, 3000);
    }
    else{
      glFrustum(-aspect, aspect, -1, 1, 2, 3000);
    }

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, width(), height());

    gluLookAt (cam_x, cam_y, eyeZ, cam_x, cam_y, cam_z, 0, -1, 0);

    glRotatef(xRot / 16.0, 1.0, 0.0, 0.0);
    glRotatef(yRot / 16.0, 0.0, 1.0, 0.0);
    glRotatef(zRot / 16.0, 0.0, 0.0, 1.0);

    // std::cout << "scale " << scale << std::endl;
    
    if(orthoProjection){
      glScalef(scale * .032 / metaParam->displayScale[0], scale * .032 / metaParam->displayScale[0], scale * .032 / metaParam->displayScale[0]);
    }
    else{
      glScalef(scale / metaParam->displayScale[0], scale / metaParam->displayScale[0], scale / metaParam->displayScale[0]);
    }
    

    // glTranslatef(0.0, 0.0, 0.0);
    glTranslatef(-centralPoint.x, -centralPoint.y, -centralPoint.z );



    if(mg_buf){

      

  		#ifdef CUDA_VBO
  			renderVBO();	
      #else
        numCells = mg_buf->numCells;
      #endif

  		// // draw grid boxes
  		// glColor3f(.80, .6, 1.0);
  		// for(uint k=0;k<mg_buf->gridSize;k++){
  		//     for(uint j=0;j<mg_buf->gridSize;j++){
  		//         for(uint i=0;i<mg_buf->gridSize;i++){

  		//             glPushMatrix();
  		//             glTranslatef(   mg_buf->worldOrigin + i * mg_buf->gridBoxSize + .5 * mg_buf->gridBoxSize,
  		//                             mg_buf->worldOrigin + j * mg_buf->gridBoxSize + .5 * mg_buf->gridBoxSize,
  		//                             mg_buf->worldOrigin + k * mg_buf->gridBoxSize + .5 * mg_buf->gridBoxSize);
  		//             glutWireCube(mg_buf->gridBoxSize);
  		//             glPopMatrix();

  		//         }
  		//     }
  		// }

  		//draw cell metric neighborhood links
  		// glColor3f(.40,1.0,.0);
  		// glLineWidth(1);
  		// glBegin(GL_LINES);
  		// for(uint i=0; i < mg_buf->numCells; i++ ){
  		//     for(uint j=0; j < mg_buf->cellMetricNeighbNum[i]; j++ ){
  		//         uint neighbId = mg_buf->cellMetricNeighbId[i * metaParam->numNeighbMax[0] + j];
  		//         glVertex3dv((double *)&(mg_buf->cellPosition[i]));
  		//         glVertex3dv((double *)&(mg_buf->cellPosition[neighbId]));
  		//     }
  		// }
  		// glEnd();

  		//draw cell topo neighborhood links
  		glColor3f(.20,.20,.20);
  		glLineWidth(2);
  		glBegin(GL_LINES);
  		bool drawwhite = false;

  		for(uint i=0; i < numCells; i++ ){

        if(slice(mg_buf->cellPosition[i])){

    		  for(uint j=0; j < mg_buf->cellTopologicalNeighbNum[i]; j++ ){
    		    uint neighbId = mg_buf->cellTopologicalNeighbId[i * NUMNEIGHBTOPOmax + j];

            if(slice(mg_buf->cellPosition[neighbId])){
      		    glVertex3dv((double *)&(mg_buf->cellPosition[i]));
      		    glVertex3dv((double *)&(mg_buf->cellPosition[neighbId]));
            } 
    		  }
  		  }
      }
  		glEnd();

  		// draw cell axes      
      if(displayParams.drawAxes){
    		glColor3f(1.0,.0,.0);
    		glLineWidth(3);
    		
    		for(uint i=0; i < numCells; i++ ){

          if(slice(mg_buf->cellPosition[i])){

            d3 currAxe(.0);
            if(displayParams.candAxesId != -1){
              currAxe = mg_buf->cellCandidateAxes[i*NUMAXESmax+displayParams.candAxesId];
            }
            else if(displayParams.axeAB != -1){
              currAxe = mg_buf->cellAxisAB[i];
            }
            else if(displayParams.axe1 != -1){
              currAxe = mg_buf->cellAxis1[i];
            }

            glColor3f(1.0,.0,.0);
            glBegin(GL_LINES);
            glVertex3dv((double *)&(mg_buf->cellPosition[i]));
            mg::d3 axis_head = mg_buf->cellPosition[i] + .03 * currAxe;
            glVertex3d(axis_head.x,axis_head.y,axis_head.z);
            glEnd();
          }
    		}
  		}

      //draw cell centers
      #ifndef CUDA_VBO

      for(uint i=0; i < numCells; i++ ){

        if(slice(mg_buf->cellPosition[i])){

          //color
          if(displayParams.currentProtein != -1){
            double nullcolor = .9;    
            double ratio = mg_buf->cellProtein[i*NUMPROTEINmax+displayParams.currentProtein]/displayParams.proteinThreshold[displayParams.currentProtein];
            glColor3f(  displayParams.proteinColorR[displayParams.currentProtein] *ratio + (1.0-ratio)*nullcolor, 
                        displayParams.proteinColorG[displayParams.currentProtein] *ratio + (1.0-ratio)*nullcolor,
                        displayParams.proteinColorB[displayParams.currentProtein] *ratio + (1.0-ratio)*nullcolor);
          }
          else if(displayParams.currentLigand != -1){
            double nullcolor = .9;    
            double ratio = mg_buf->cellLigand[i*NUMLIGmax+displayParams.currentLigand]/displayParams.ligandThreshold[displayParams.currentLigand];
            glColor3f(  displayParams.ligandColorR[displayParams.currentLigand] *ratio + (1.0-ratio)*nullcolor, 
                        displayParams.ligandColorG[displayParams.currentLigand] *ratio + (1.0-ratio)*nullcolor,
                        displayParams.ligandColorB[displayParams.currentLigand] *ratio + (1.0-ratio)*nullcolor);
          }
          else if(displayParams.colorByType == 1){

            uint celltype = mg_buf->cellType[i];
            uint epiPolarized = mg_buf->cellEpiIsPolarized[i];

            if(celltype == 2 && epiPolarized == 1){
              celltype = 3;
            }

            glColor3f( displayParams.typeColorR[celltype], 
                            displayParams.typeColorG[celltype],
                            displayParams.typeColorB[celltype] );

            // //Boundary Formation
            uint cellipid = mg_buf->cellEpiId[i];
            if(cellipid == 5){
              glColor3f(110.0/255.0, 227.0/255.0, 28.0/255.0);
            }
            else if(cellipid == 8){
              glColor3f(255.0/255.0, 66.0/255.0, 0.0/255.0);
            }
          }
          else{
            glColor3f(displayParams.noneColorR, displayParams.noneColorG, displayParams.noneColorB);
          }
          /// XXX version papier epiboly
          // glColor3f(.0, .0, .0);

          //solid sphere
          glPushMatrix();
          glTranslated(mg_buf->cellPosition[i].x,mg_buf->cellPosition[i].y,mg_buf->cellPosition[i].z);
          
          // glutSolidSphere(.5*mg_buf->cellRadius[i].x,10,10); 
          
          GLUquadricObj *quadObj = gluNewQuadric();
          gluQuadricDrawStyle(quadObj, GLU_FILL);
          gluQuadricNormals(quadObj, GLU_SMOOTH);
          gluSphere(quadObj, .5*mg_buf->cellRadius[i].x, 10, 10);
          gluDeleteQuadric(quadObj);
          
          glPopMatrix();

          //XXX version papier epiboly
          // for(uint j=0; j < mg_buf->cellTopologicalNeighbNum[i]; j++ ){
          //     uint neighbId = mg_buf->cellTopologicalNeighbId[i * NUMNEIGHBTOPOmax + j];
          //     glPushMatrix();
          //     glTranslated(mg_buf->cellPosition[neighbId].x,mg_buf->cellPosition[neighbId].y,mg_buf->cellPosition[neighbId].z);
          //     glutSolidSphere(.01,10,10); 
          //     glPopMatrix();  
          // }

          //vertex
          // glPointSize(33.0);
          // glBegin(GL_POINTS);
          // glVertex3dv((double *)&(mg_buf->cellPosition[i]));
          // glEnd();
        }
      }
      #endif

      // macro replaced by custom code in customdisplay.hpp
      PRINTCUSTOM

  		//2D display
      glEnable(GL_DEPTH_TEST);
  		GLfloat projection[16];
  		GLfloat modelview[16];
  		glGetFloatv( GL_PROJECTION_MATRIX, projection );
  		glGetFloatv( GL_MODELVIEW_MATRIX, modelview );

  		//printf hour TS / hpf
  		glMatrixMode(GL_PROJECTION);
  		glLoadIdentity();
  		glOrtho(0, width(), 0,height(), -1, 1);

  		glMatrixMode(GL_MODELVIEW);
  		glLoadIdentity();

      glColor3f (.0,.0,.0);
  		// glColor3f (1.0,1.0,1.0);
  		char c[256];
  		glRasterPos2f(20, 20);

  		sprintf(c,"T = %3.2lf hpf / %d TS / %d cells",
  		    (mg_buf->TS * param->deltaTime[0] / 3600.0) + 4.0, mg_buf->TS, numCells);
  		// glPrint(c);

  		int i=0;
  		while(c[i] != '\0'){
  		  glutBitmapCharacter(GLUT_BITMAP_9_BY_15,c[i]);
  		  i++;
  		}

  		// draw cell id
    	if(twoDinfo){

      	// char c[32];
    		glColor3f (.0,1.0,.0);
    		for(uint i=0; i< numCells; i++){

  		    d3 pos = mg_buf->cellPosition[i];

          if(slice(pos)){

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

  		        double px = width() * (1.0 + o.x) / 2.0;
  		        double py = height() * (1.0 + o.y) / 2.0;

  		        glRasterPos2f(px,py);
              // sprintf(c,"%d",i);
              // sprintf(c,"%d %.3lf",i,mg_buf->cellLigand[i*NUMLIGmax+0]);
              sprintf(c,"%d ",i);

              if(displayParams.currentProtein != -1){
                double qmol = mg_buf->cellProtein[i*NUMPROTEINmax+displayParams.currentProtein];
                sprintf(c, "%s %.3lf", c, qmol);
              }
              else if(displayParams.currentLigand != -1){
                double qmol = mg_buf->cellLigand[i*NUMLIGmax+displayParams.currentLigand];
                sprintf(c, "%s %.3lf", c, qmol);
              }

  		        // sprintf(c,"%d %.3lf",i,qmol);
  		        // glPrint(cVal);
  		        int id=0;
  		        while(c[id] != '\0'){
		            glutBitmapCharacter(GLUT_BITMAP_9_BY_15,c[id]);
		            id++;
  		        }
  		      }
          }
    		}

        PRINTCUSTOM2D
      }

      //recording stuff
      if(recording || record_single_snapshot){

        if(unrecorded){
        // if(mg_buf->TS%recording_period == 0 && unrecorded){

          
          saveScreenToPNG(recording_timer, "image_");
          recording_timer++;
          unrecorded = false;
          unlockConsumer();
        }
        else if(record_single_snapshot){
          saveScreenToPNG(snapshot_counter, "snapshot_");
          snapshot_counter++;
          record_single_snapshot=false;
        }

	    }
	  }
    glFlush();
  }

  uint GLWidget::slice(d3 pos){
      
    double scal = dot(pos, LRaxis);
 
    if( scal > SlicerYmax || scal < SlicerYmin){
      return 0;
    }
    
    scal = dot(pos, APaxis);
            
    if( scal > SlicerXmax || scal < SlicerXmin){
      return 0;
    }

    scal = dot(pos, AVaxis); 

    if( scal > SlicerZmax || scal < SlicerZmin){
      return 0;
    }

    return 1;
  }

  void GLWidget::pickaCenter(){
  

    double w = width();
    double h = height();
    double aspect = w/h;
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    if(orthoProjection){
      glOrtho(-aspect, aspect, -1, 1, 2, 3000);
    }
    else{
      glFrustum(-aspect, aspect, -1, 1, 2, 3000);
    }
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, width(), height());
    
    gluLookAt (cam_x, cam_y, eyeZ,
      cam_x, cam_y, cam_z,
      0, -1, 0);

    glRotatef(xRot / 16.0, 1.0, 0.0, 0.0);
    glRotatef(yRot / 16.0, 0.0, 1.0, 0.0);
    glRotatef(zRot / 16.0, 0.0, 0.0, 1.0);



    if(orthoProjection){
      glScalef(scale * .032 / metaParam->displayScale[0], scale * .032 / metaParam->displayScale[0], scale * .032 / metaParam->displayScale[0]);
    }
    else{
      glScalef(scale / metaParam->displayScale[0], scale / metaParam->displayScale[0], scale / metaParam->displayScale[0]);
    }
    
    glTranslatef(-centralPoint.x, -centralPoint.y, -centralPoint.z );
    
    glGetFloatv( GL_PROJECTION_MATRIX, projection );
    glGetFloatv( GL_MODELVIEW_MATRIX, modelview );
    
    double x,y,z,ax,ay,az,aw,ox,oy,ow,px,py;  
    
    for(uint i=0; i < numCells; i++){

      if(slice(mg_buf->cellPosition[i])){
      
        x = mg_buf->cellPosition[i].x;
        y = mg_buf->cellPosition[i].y;
        z = mg_buf->cellPosition[i].z;
        
        ax = modelview[0]*x + modelview[4]*y + modelview[8]*z + modelview[12];
        ay = modelview[1]*x + modelview[5]*y + modelview[9]*z + modelview[13];
        az = modelview[2]*x + modelview[6]*y + modelview[10]*z + modelview[14];
        aw = modelview[3]*x + modelview[7]*y + modelview[11]*z + modelview[15];

        ox = projection[0]*ax + projection[4]*ay + projection[8]*az + projection[12]*aw;  
        oy = projection[1]*ax + projection[5]*ay + projection[9]*az + projection[13]*aw;
        ow = projection[3]*ax + projection[7]*ay + projection[11]*az + projection[15]*aw;

        if (ow != 0.0) ox /= ow;
        px = width() * (1.0 + ox) / 2.0;
        
        if (ow != 0.0) oy /= ow;
        py = height() * (1.0 + oy) / 2.0;
      
        if( (px-mouseX)*(px-mouseX) + (py-mouseY)*(py-mouseY) < 20 ){
          centralPoint = mg_buf->cellPosition[i];  
          printf("cell %d is center\n",i);
        }
      
      }
    }     

  }

  void GLWidget::saveScreenToPNG(int counter, QString image_name){
    QImage qimage  = grabFrameBuffer(0);
    char adr[400];
    sprintf(    
                adr,
                "%s/%s%05d.png", 
                recording_folder.toLocal8Bit().data(), 
                image_name.toLocal8Bit().data(), 
                counter
            );
    std::cout << "Recording display to " << adr << std::endl;
    qimage.save(QString(adr),0,100);
  }

  void GLWidget::resizeGL(int width, int height)
  {
    int side = qMin(width, height);
    glViewport((width - side) / 2, (height - side) / 2, side, side);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
#ifdef QT_OPENGL_ES_1
    glOrthof(-0.5, +0.5, -0.5, +0.5, 4.0, 15.0);
#else
    glOrtho(-0.5, +0.5, -0.5, +0.5, 4.0, 15.0);
#endif
    glMatrixMode(GL_MODELVIEW);
  }

  void GLWidget::unlockConsumer(){
    //unlock waiting consumer
    isf::Autolock v(mutex);
    cv.signal();
  }

  //
  // Data related:
  //

  void GLWidget::setBuffer(mg::EmbryoState * buf, int step) {
    mg_buf = buf;
    #ifdef CUDA_VBO
        newDataAvailable = 1;
    #endif

    //blocking condition if glwidget is recording snapshot
    if(recording == true){
      //wait unlocking if data time step must be recorded 
      if(step%recording_period == 0){
        unrecorded = true;
        isf::Autolock v(mutex);
        cv.wait(mutex);
      }
    }

  }

#ifdef CUDA_VBO

  //
  // VBO/Data initialization:
  //

  void GLWidget::initShaders()
  {
    m_program = _compileProgram(
        const_cast<char *>("./gui/data/mecaviz.vert"),
        const_cast<char *>("./gui/data/mecaviz.frag") 
		);
  }

  void GLWidget::initData()
  {

    // load sphere data:
    float * m_hSphereVertex = new float [3 * NUMSPHEREVERTEX];    // 3 floats/vertex
    uint *  m_hSphereIndice = new uint  [3 * NUMSPHERETRIANGLE];  // 3 vertexes/triangle

    uint * SphereVertexNumTriVois_h = new uint[NUMSPHEREVERTEX];
    uint * SphereVertexIDTriVois_h  = new uint[10 * NUMSPHEREVERTEX];
    memset(SphereVertexNumTriVois_h, 0, NUMSPHEREVERTEX * sizeof(uint));

    // Data initialization:
    // Override system locale until shaders are compiled
    setlocale(LC_NUMERIC, "C");
    FILE *file = fopen("./gui/data/icosphere3.obj", "r");
    char letter [80];
    if(file==NULL) { printf("Error: can't open sphere data.\n"); }
    else {
      int scan;
      // Read the vertexes from the file:
      for(int i=0;i<NUMSPHEREVERTEX;i++){
        scan=fscanf(file, "%s",letter);
        scan=fscanf(file, "%f",&m_hSphereVertex[3 * i + 0]);
        scan=fscanf(file, "%f",&m_hSphereVertex[3 * i + 1]);
        scan=fscanf(file, "%f",&m_hSphereVertex[3 * i + 2]);
      }
      // Read the triangles from the file:
      for(int i=0;i<NUMSPHERETRIANGLE;i++){
        scan=fscanf(file, "%s",letter);
        scan=fscanf(file, "%d",&m_hSphereIndice[3 * i + 0]);
        scan=fscanf(file, "%d",&m_hSphereIndice[3 * i + 1]);
        scan=fscanf(file, "%d",&m_hSphereIndice[3 * i + 2]);

        // Indices are stored starting to 1: convert to array indices.
        m_hSphereIndice[3 * i + 0]--;
        m_hSphereIndice[3 * i + 1]--;
        m_hSphereIndice[3 * i + 2]--;

        SphereVertexIDTriVois_h[10*m_hSphereIndice[3 * i + 0] + SphereVertexNumTriVois_h[m_hSphereIndice[3 * i + 0]] ] = i;
        SphereVertexNumTriVois_h[m_hSphereIndice[3 * i + 0]]++;
        SphereVertexIDTriVois_h[10*m_hSphereIndice[3 * i + 1]+ SphereVertexNumTriVois_h[m_hSphereIndice[3 * i + 1]] ] = i;
        SphereVertexNumTriVois_h[m_hSphereIndice[3 * i + 1]]++;
        SphereVertexIDTriVois_h[10*m_hSphereIndice[3 * i + 2]+ SphereVertexNumTriVois_h[m_hSphereIndice[3 * i + 2]] ] = i;
        SphereVertexNumTriVois_h[m_hSphereIndice[3 * i + 2]]++;
      }
      fclose(file);
    }

    // Restore system locale
    setlocale(LC_ALL, "");

    // Normalize sphere: Compute the barycentre
    f3 center = f3(.0f);
    for(uint i=0; i<NUMSPHEREVERTEX; i++){
      center.x += m_hSphereVertex[3*i+0];
      center.y += m_hSphereVertex[3*i+1];
      center.z += m_hSphereVertex[3*i+2];
    }
    center /= (float)NUMSPHEREVERTEX;
    // Normalize sphere: Recenter
    for(uint i=0;i<NUMSPHEREVERTEX;i++){
      m_hSphereVertex[3*i+0] -= center.x;
      m_hSphereVertex[3*i+1] -= center.y;
      m_hSphereVertex[3*i+2] -= center.z;
    }
    // Compute farther vertex from (0,0,0)
    float distmax = .0f, dist;
    for(uint i=0; i<NUMSPHEREVERTEX; i++){
      dist = (float) sqrt(
          m_hSphereVertex[3*i+0] * m_hSphereVertex[3*i+0] +
          m_hSphereVertex[3*i+1] * m_hSphereVertex[3*i+1] +
          m_hSphereVertex[3*i+2] * m_hSphereVertex[3*i+2]
          );
      if(dist > distmax){ distmax = dist; }
    }
    // Rescale all point:
    for(uint i=0; i<(NUMSPHEREVERTEX*3); i++){ m_hSphereVertex[i] /= distmax; }

    //
    // Create VBOs
    //

    GLuint vbo;
    // Create vertex VBO:  Number of VBO variables/vertex * number of vertexes/sphere * number of cells
    float * m_hvertexes = new float[NUMVBOVAR * NUMSPHEREVERTEX * NUMPARTINNERCELLmax];
    for(uint i = 0; i < NUMSPHEREVERTEX * NUMPARTINNERCELLmax; ++i){
      m_hvertexes[NUMVBOVAR*i+0] = ((float)rand()/(float)RAND_MAX)*200-100;
      m_hvertexes[NUMVBOVAR*i+1] = ((float)rand()/(float)RAND_MAX)*200-100;
      m_hvertexes[NUMVBOVAR*i+2] = ((float)rand()/(float)RAND_MAX)*200-100;
      m_hvertexes[NUMVBOVAR*i+3] = 1.0;
      m_hvertexes[NUMVBOVAR*i+4] = 0.0;
      m_hvertexes[NUMVBOVAR*i+5] = 0.0;
      m_hvertexes[NUMVBOVAR*i+6] = ((float)rand()/(float)RAND_MAX)*200-100;
      m_hvertexes[NUMVBOVAR*i+7] = ((float)rand()/(float)RAND_MAX)*200-100;
      m_hvertexes[NUMVBOVAR*i+8] = ((float)rand()/(float)RAND_MAX)*200-100;
      m_hvertexes[NUMVBOVAR*i+9] = 0.0;
    }
    // Create the vbo with the datas
    glGenBuffers(1, &posVBO_id);
    glBindBuffer(GL_ARRAY_BUFFER, posVBO_id);
    glBufferData(GL_ARRAY_BUFFER, NUMVBOVAR * NUMSPHEREVERTEX * NUMPARTINNERCELLmax * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    registerGLBufferObject(posVBO_id, &(cuda_posVBO_resource));

    // Create triangle (element) VBO:
    // Triangle are 3 indices to vertexes, we precompute all indices, even if we do not have the vertexes yet.
    uint * m_hIndices = new uint[NUMSPHERETRIANGLE * NUMPARTINNERCELLmax * 3];
    for(uint i=0; i<NUMPARTINNERCELLmax; i++){
      for(uint j=0; j<NUMSPHERETRIANGLE; j++){
        m_hIndices[3 * (i * NUMSPHERETRIANGLE + j) + 0] = i * NUMSPHEREVERTEX + m_hSphereIndice[3*j + 0];
        m_hIndices[3 * (i * NUMSPHERETRIANGLE + j) + 1] = i * NUMSPHEREVERTEX + m_hSphereIndice[3*j + 1];
        m_hIndices[3 * (i * NUMSPHERETRIANGLE + j) + 2] = i * NUMSPHEREVERTEX + m_hSphereIndice[3*j + 2];
      }
    }
    // Create the vbo with the datas
    glGenBuffers(1, &elementVBO_id);                      //Create a new VBO and use the variable id to store the VBO id
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementVBO_id);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, NUMSPHERETRIANGLE * NUMPARTINNERCELLmax * 3 * sizeof(uint), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementVBO_id);
    glBufferSubData(
        GL_ELEMENT_ARRAY_BUFFER, 
        0,  //offset in bytes : cells data
        3 * NUMSPHERETRIANGLE * NUMPARTINNERCELLmax * sizeof(uint), 
        (const GLvoid*)m_hIndices
        );
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    allocateArray((void**)&SphereVertex_d,            3   * NUMSPHEREVERTEX     * sizeof(float));
    allocateArray((void**)&SphereIndice_d,            3   * NUMSPHERETRIANGLE   * sizeof(uint));
    allocateArray((void**)&SphereVertexNumTriVois_d,        NUMSPHEREVERTEX     * sizeof(uint));
    allocateArray((void**)&SphereVertexIDTriVois_d,   10  * NUMSPHEREVERTEX     * sizeof(uint));

    copyArrayToDevice(SphereVertex_d, m_hSphereVertex, 0, 3 * NUMSPHEREVERTEX *sizeof(float));
    copyArrayToDevice(SphereIndice_d, m_hSphereIndice, 0, 3 * NUMSPHERETRIANGLE * sizeof(uint));
    copyArrayToDevice(SphereVertexNumTriVois_d, SphereVertexNumTriVois_h, 0, NUMSPHEREVERTEX * sizeof(uint));
    copyArrayToDevice(SphereVertexIDTriVois_d, SphereVertexIDTriVois_h, 0, 10 * NUMSPHEREVERTEX * sizeof(uint));

    allocateArray((void**)&cellTriangleNormales_d, NUMSPHERETRIANGLE * NUMPARTINNERCELLmax *sizeof(d3));

    // Free
    delete [] m_hIndices;
    delete [] m_hvertexes;
    delete [] m_hSphereVertex;
    delete [] m_hSphereIndice;
    delete [] SphereVertexNumTriVois_h;
    delete [] SphereVertexIDTriVois_h;

    // Other allocations, fill-in later:
    // cellPosition_d = 0;
    allocateArray((void**)&cellPosition_d,                  NUMPARTINNERCELLmax * sizeof(d3));
    allocateArray((void**)&cellRadius_d,                    NUMPARTINNERCELLmax * sizeof(d3));
    allocateArray((void**)&cellAxis_d,                      NUMPARTINNERCELLmax * sizeof(d3));
    allocateArray((void**)&cellTopologicalNeighbNum_d,      NUMPARTINNERCELLmax * sizeof(uint));
    allocateArray((void**)&cellTopologicalNeighbId_d,       NUMPARTINNERCELLmax * NUMNEIGHBTOPOmax * sizeof(uint));
    allocateArray((void**)&cellPopulation_d,                NUMPARTINNERCELLmax * sizeof(uint));
    allocateArray((void**)&cellType_d,                      NUMPARTINNERCELLmax * sizeof(uint));
    allocateArray((void**)&cellEpiIsPolarized_d,            NUMPARTINNERCELLmax * sizeof(uint));
    allocateArray((void**)&cellEpiId_d,                     NUMPARTINNERCELLmax * sizeof(uint));
    allocateArray((void**)&cellLigand_d,                    NUMPARTINNERCELLmax * NUMLIGmax * sizeof(double));
    allocateArray((void**)&cellProtein_d,                   NUMPARTINNERCELLmax * NUMPROTEINmax * sizeof(double));

  }

  //
  // VBO/Computation:
  //

  void GLWidget::updateDeviceArrays()
  {

   
    copyArrayToDevice(cellTopologicalNeighbNum_d, mg_buf->cellTopologicalNeighbNum, 0, numCells *sizeof(uint));
    copyArrayToDevice(cellTopologicalNeighbId_d, mg_buf->cellTopologicalNeighbId, 0, numCells *NUMNEIGHBTOPOmax*sizeof(uint));
    copyArrayToDevice(cellPopulation_d, mg_buf->cellPopulation, 0, numCells * sizeof(uint));
    copyArrayToDevice(cellType_d, mg_buf->cellType, 0, numCells * sizeof(uint));
    copyArrayToDevice(cellEpiIsPolarized_d, mg_buf->cellEpiIsPolarized, 0, numCells * sizeof(uint));
    copyArrayToDevice(cellEpiId_d, mg_buf->cellEpiId, 0, numCells * sizeof(uint));


    //cuda capabilities version 1.3+
    copyArrayToDevice(cellPosition_d,             mg_buf->cellPosition,             0, mg_buf->numCells *sizeof(d3));
   	copyArrayToDevice(cellRadius_d,               mg_buf->cellRadius,               0, mg_buf->numCells *sizeof(d3));
    copyArrayToDevice(cellAxis_d,                 mg_buf->cellAxisAB,               0, mg_buf->numCells *sizeof(d3));
    copyArrayToDevice(cellLigand_d,               mg_buf->cellLigand,               0, mg_buf->numCells * NUMLIGmax * sizeof(double));
    copyArrayToDevice(cellProtein_d,               mg_buf->cellProtein,               0, mg_buf->numCells * NUMPROTEINmax * sizeof(double));
    
    // Map the openGL buffer to a cuda usable array and lock the memory area
    float * output_vbo  = (float *) mapGLBufferObject(&cuda_posVBO_resource);

    // // Perform the cuda call
    update_vbo(
        &displayParams,
        output_vbo,                       // Where to write on the GPU
        numCells,                 // number of cells
        cellPosition_d,                   // cell position
        cellRadius_d,                     // cell radius
        cellTopologicalNeighbNum_d,
        cellTopologicalNeighbId_d,
        cellAxis_d,                       // Cell axis
        SphereVertex_d,                   // Sphere vertex
        SphereIndice_d,                   // Sphere indice
        metaParam->numNeighbMax[0],
        SphereVertexNumTriVois_d,
        SphereVertexIDTriVois_d,
        cellPopulation_d,
        cellLigand_d,
        cellProtein_d,
        sea_urchin_pop_colors[0],
        sea_urchin_pop_colors[1],
        sea_urchin_pop_colors[2],
        sea_urchin_pop_colors[3],
        cellTriangleNormales_d,
        cellType_d,
        cellEpiIsPolarized_d,
        cellEpiId_d
        );

    // // Unmap the openGL buffer from cuda and unlock
    unmapGLBufferObject(cuda_posVBO_resource);   
  }

  //
  // VBO/Rendering:
  //

  void GLWidget::renderVBO(){

    //if(mg_buf){

    if(newDataAvailable.testAndSetOrdered(1,0)){
      numCells = mg_buf->numCells;
      updateDeviceArrays();
      // newDataAvailable = 0;
    }
    else if(newParamAvailable.testAndSetOrdered(1,0)){
      updateDeviceArrays();
      // newParamAvailable = 0;
    }

    //clipping plane
    double plane0Eq[4] = {LRaxis.x, LRaxis.y, LRaxis.z, -SlicerYmin};
    glClipPlane(GL_CLIP_PLANE0, plane0Eq);
    glEnable(GL_CLIP_PLANE0); 
    double plane1Eq[4] = {-LRaxis.x, -LRaxis.y, -LRaxis.z, SlicerYmax};
    glClipPlane(GL_CLIP_PLANE1, plane1Eq);
    glEnable(GL_CLIP_PLANE1); 
    double plane2Eq[4] = {APaxis.x, APaxis.y, APaxis.z, -SlicerXmin};
    glClipPlane(GL_CLIP_PLANE2, plane2Eq);
    glEnable(GL_CLIP_PLANE2); 
    double plane3Eq[4] = {-APaxis.x, -APaxis.y, -APaxis.z, SlicerXmax};
    glClipPlane(GL_CLIP_PLANE3, plane3Eq);
    glEnable(GL_CLIP_PLANE3); 
    double plane4Eq[4] = {AVaxis.x, AVaxis.y, AVaxis.z, -SlicerZmin};
    glClipPlane(GL_CLIP_PLANE4, plane4Eq);
    glEnable(GL_CLIP_PLANE4); 
    double plane5Eq[4] = {-AVaxis.x, -AVaxis.y, -AVaxis.z, SlicerZmax};
    glClipPlane(GL_CLIP_PLANE5, plane5Eq);
    glEnable(GL_CLIP_PLANE5); 

      


    // Draw
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(m_program);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, posVBO_id);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, posVBO_id);

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, NUMVBOVAR * sizeof(float), (char *)NULL);

    glEnableClientState(GL_NORMAL_ARRAY);
    glNormalPointer(GL_FLOAT, NUMVBOVAR * sizeof(float), ((char *)NULL + (12)));

    GLint loc = glGetAttribLocation(m_program,"scalarR");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc,1,GL_FLOAT,0, NUMVBOVAR * sizeof(float),((char *)NULL + (24)));
    GLint loc1 = glGetAttribLocation(m_program,"scalarG");
    glEnableVertexAttribArray(loc1);
    glVertexAttribPointer(loc1,1,GL_FLOAT,0, NUMVBOVAR * sizeof(float),((char *)NULL + (28)));
    GLint loc2 = glGetAttribLocation(m_program,"scalarB");
    glEnableVertexAttribArray(loc2);
    glVertexAttribPointer(loc2,1,GL_FLOAT,0, NUMVBOVAR * sizeof(float),((char *)NULL + (32)));

    GLint loc3 = glGetAttribLocation(m_program,"hide");
    glEnableVertexAttribArray(loc3);
    glVertexAttribPointer(loc3,1,GL_FLOAT,0, NUMVBOVAR * sizeof(float),((char *)NULL + (36)));

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, elementVBO_id);

    glDrawElements(GL_TRIANGLES, 3 * numCells * NUMSPHERETRIANGLE,
        GL_UNSIGNED_INT, 0);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    glUseProgram(0);
    glDisable(GL_BLEND);
    glDisable(GL_LIGHTING);

    //uncomment to superpose VBO and nonVBO rendering
    // glDisable(GL_DEPTH_TEST);
    glDisable(GL_CLIP_PLANE0);
    glDisable(GL_CLIP_PLANE1);
    glDisable(GL_CLIP_PLANE2);
    glDisable(GL_CLIP_PLANE3);
    glDisable(GL_CLIP_PLANE4);
    glDisable(GL_CLIP_PLANE5);
    //}
  }

  //
  // VBO/Helpers:
  //

  GLuint GLWidget::_compileProgram(char *vsource, char *fsource)
  {
    GLint compile_status;
    char  log[256];

    // Create the structures:
    GLuint vertexShader   = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    // Read the files
    char * vs = textFileRead(vsource);
    char * fs = textFileRead(fsource);

    // Compile
    const char * vv = vs;
    const char * ff = fs;
    glShaderSource(vertexShader, 1, &vv, 0);
    glShaderSource(fragmentShader, 1, &ff, 0);
    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    // Check if the compilation of the vertex shader succeed
    compile_status = GL_TRUE;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &compile_status);
    if(compile_status != GL_TRUE)
    {
      glGetShaderInfoLog(vertexShader, 256, 0, log);
      fprintf(stderr, "impossible de compiler le vertex shader  :\n%s\n", log);
      exit(1);
    }

    // Check if the compilation of the fragment shader succeed
    compile_status = GL_TRUE;
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &compile_status);
    if(compile_status != GL_TRUE)
    {
      glGetShaderInfoLog(fragmentShader, 256, 0, log);
      fprintf(stderr, "impossible de compiler le pixel shader  :\n%s\n", log);
      exit(1);
    }

    // Create the program:
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
      char temp[256];
      glGetProgramInfoLog(program, 256, 0, temp);
      printf("Failed to link program:\n%s\n", temp);
      glDeleteProgram(program);
      program = 1;
    }

    return program;
  }

  char * GLWidget::textFileRead(char *fn) {
    FILE *fp;
    char *content = NULL;
    int count=0;
    if (fn != NULL) {
      fp = fopen(fn,"rt");
      if (fp != NULL) {
        fseek(fp, 0, SEEK_END);
        count = ftell(fp);
        rewind(fp);
        if (count > 0) {
          content = (char *)malloc(sizeof(char) * (count+1));
          count = fread(content,sizeof(char),count,fp);
          content[count] = '\0';
        }
        fclose(fp);
      }
    }
    return content;
  }

#endif

}
