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

#ifndef GLWIDGETVBO_H
#define GLWIDGETVBO_H
// Creating our own QGLWidget in order to display the urchin

// Standard includes:
#include <math.h>

// OpenGL:
#include "GL/glew.h"
#include <GL/glut.h>

// Qt includes:
#include <QGLWidget>
#include <QTimer>
#include <QMouseEvent>
#include <QAtomicInt>

// Project includes:
#include "thrust_objects.hpp"
#include "metaparam.hpp"
#include "param.hpp"
#include "cells_producer.hpp"
#include "define_gui.h"
#include "displayparameters.hpp"

#include "SDL_customized_objects.h"

#ifdef CUDA_VBO
struct cudaGraphicsResource;
#endif

namespace mg{

  struct EmbryoState;
  template<int T> class MetaParam;
  template<int T> class Param;

  // class GLWidget : public QGLWidget, public QOpenGLFunctions
  class GLWidget : public QGLWidget
  {
    Q_OBJECT

    public:
      explicit GLWidget(MetaParam<HOST> * _mp, Param<HOST> * _p, QGLFormat format, QWidget *parent = 0 );
      ~GLWidget();

      QSize minimumSizeHint() const;
      QSize sizeHint()        const;

      void setBuffer(EmbryoState * buf, int step);
      
      // TODO: private + fonction
      double  scale;

      // OpenGL camera related:
      int     xRot, yRot, zRot;
      double  cam_x, cam_y, cam_z;
      double  eyeZ;
      QPoint lastPos;
      bool orthoProjection;
      bool twoDinfo;

      //Embryo axes 
      d3                LRaxis;
      d3                APaxis;
      d3                AVaxis;
      float             SlicerXmax;
      float             SlicerYmax;
      float             SlicerZmax;
      float             SlicerXmin;
      float             SlicerYmin;
      float             SlicerZmin;

      d3                centralPoint;

      //recording stuff
      bool recording;
      isf::Mutex mutex;
      isf::ConditionVariable cv;
      int recording_period;
      int recording_timer;
      QString recording_folder;
      bool unrecorded;

      bool record_single_snapshot;
      int snapshot_counter;

      //GRN display
      // f3                proteinColor[NUMPROTEINmax];
      // float             proteinThreshold[NUMPROTEINmax];
      // int               currentProtein;
      // f3                ligandColor[NUMLIGmax];
      // float             ligandThreshold[NUMLIGmax];
      // int               currentLigand;
      DisplayParams        displayParams;

      void unlockConsumer();
      uint slice(d3 pos);
      void pickaCenter();

    protected:
      void mousePressEvent(QMouseEvent *e);
      void mouseMoveEvent(QMouseEvent *event);
      void timerEvent(QTimerEvent *e);

      void initializeGL();
      void resizeGL(int width, int height);
      void paintGL();

    private:
      // OpenGL painting related:
      QTimer           timer;
      // Project data
      MetaParam<HOST> * metaParam;
      Param<HOST> *     param;
      EmbryoState *     mg_buf;
      f3                sea_urchin_pop_colors[4];
	    uint 				      numCells;

      void saveScreenToPNG(int counter, QString image_name);
      
      
      // Qt communication:
      public slots:
      void setXRotation(int angle);
      void setYRotation(int angle);
      void setZRotation(int angle);
      void setSlicerXmax(int i);
      void setSlicerXmin(int i);
      void setSlicerYmax(int i);
      void setSlicerYmin(int i);
      void setSlicerZmax(int i);
      void setSlicerZmin(int i);
      void setOrthoProj();
      void setPerspProj();
      void set2Dinfo(bool on);
      void setColByType(bool on);
      void newDisplayParamVBO();
      void setDrawAxes(bool on);
      void setAxesId(int i);

    signals:
      void xRotationChanged(int angle);
      void yRotationChanged(int angle);
      void zRotationChanged(int angle);




#ifdef CUDA_VBO

    public:

      // Printing control:
      QAtomicInt    newDataAvailable;
      QAtomicInt    newParamAvailable;

    private:

      // Device management:
      GLuint                        m_program;
      uint                          posVBO_id;
      uint                          elementVBO_id;
      struct cudaGraphicsResource * cuda_posVBO_resource;

      // Algorithm's data
      float * SphereVertex_d;             // Device sphere vertex (CUDA)
      uint  * SphereIndice_d;             // Device sphere indice (CUDA)
      uint  * SphereVertexNumTriVois_d;
      uint  * SphereVertexIDTriVois_d;
      d3*     cellPosition_d;
      d3*     cellRadius_d;
      uint*   cellTopologicalNeighbNum_d;
      uint*   cellTopologicalNeighbId_d;
      d3*     cellAxis_d;
      uint*   cellPopulation_d;
      uint*   cellType_d;
      uint*   cellEpiIsPolarized_d;
      uint*   cellEpiId_d;
      d3*     cellTriangleNormales_d;
      double*  cellLigand_d;
      double*  cellProtein_d;

      // f3*     cellPosition_h;
      // f3*     cellRadius_h;
      // f3*     cellAxis_h;
      // float*  cellLigand_h;

      // Initialization:
      void initShaders();
      void initData();
      // Computation:
      void updateDeviceArrays();
      // Rendering:
      void renderVBO();
      // Helpers:
      GLuint _compileProgram(char *vsource, char *fsource);
      char * textFileRead(char *fn);
#endif

  };

}

#endif

