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

#ifndef WINDOW_H
#define WINDOW_H

// OpenGL:
#include "GL/glew.h"
#include <GL/glut.h>

// Qt includes
#include <QtWidgets>    // QHBoxLayout and other widgets
#include <QWidget>      // Base Qt widget class
#include <QPushButton>
#include <QRadioButton>
#include <QCheckBox>
#include <QSlider>

// Project includes
// Thrust:
#include "thrust_objects.hpp"
// Model/simulation
#include "metaparam.hpp"
#include "param_host.hpp"
#include "state_host.hpp"
#include "model.hpp"
// OpenGL:
#include "glwidget.hpp"
#include "displaycontrolwidget.hpp"
#include "customparamwidget.hpp"
#include "recordingwidget.hpp"

namespace isf {
  template< class T1, class T2, class T3> class Controller;
}

namespace mg{

  template<int T> class MetaParam;
  class Param_Host;
  class State_Host;

  class GLWidget;

  class Window : public QWidget
  {
    Q_OBJECT

    public:
      Window( isf::Controller< MetaParam<HOST>, Param_Host, State_Host >** _c,
          MetaParam<HOST> * _mp,
          Param_Host * _p,
          State_Host * _s,
          QApplication *parent = 0
          );

      ~Window() throw ();

      void closeEvent(QCloseEvent *event);
      void initGUI(int ts, int te);

    public:
      // WidgetOGL: make this private with a friend class
      GLWidget *glWidget;
      DisplayControlWidget *disConWidget;
      CustomParamWidget *customParamWidget;
      RecordingWidget *recordingWidget;

    private:
      // Private functions:
      // QSlider * createSlider(int rmin, int rmax, int step);
      void keyPressEvent(QKeyEvent *event);
      void wheelEvent (QWheelEvent * event);
      // void initSlideêrs();

      // Gui buttons:
      QPushButton * playButton;
      QPushButton * pauseButton;
      QPushButton * plusOneButton;
      QCheckBox * disConButton;
      QRadioButton * customParamButton;
      QCheckBox * recordSnapButton;
      QPushButton * recordStateParamButton;

      // Controller to talk to:
      isf::Controller< MetaParam<HOST>, Param_Host, State_Host >** c;
      MetaParam<HOST>* metaParam;
      Param_Host*     param;
      State_Host*     state;

      // Simulation time
      int time_start;
      int time_end;

      public slots:
      // Gui buttons slots
      void callStart();
      void callPause();
      void callPlusOne();
      void showHideDisplayControl();
      void showHideCustomParam();
      void showHideRecording();
      void saveStateParam();
      
      // Gui sliders slots:
      // void setAdhHomotypic(int val);
      // void setAdhHeterotypic(int val);
      // void setRepulsion(int val);
      // void setRigidity(int val);
      // void setGabCoeff(int val);
      // void setCmax(int val);
      void setDamping(int val);
  };

}

#endif
