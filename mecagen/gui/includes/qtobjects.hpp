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

#ifndef QTOBJECTS_H
#define QTOBJECTS_H

#include <QLabel>
#include <QColorDialog>
#include <QRadioButton>
#include <QWheelEvent>

#include "displaycontrolwidget.hpp"
#include "thrust_objects.hpp"

namespace mg{

  class ColorLabel : public QLabel
  {
    Q_OBJECT

    public:
      
      ColorLabel(float* _colorR, float* _colorG, float* _colorB, QWidget *parent = 0);
      
      void mouseDoubleClickEvent( QMouseEvent * event );
    
    public slots:
      
    signals:
     void doubleClicked();
    
    private:
      float*    colorR;
      float*    colorG;
      float*    colorB;
  };

  class ProtLigRadioButton : public QRadioButton
  {
    Q_OBJECT

    public:
      
      ProtLigRadioButton(int* _currentProt, int* _currentLig, int _id, int _mode, const QString & text, QWidget *parent = 0);
      
    public slots:
      
      void clicked();

    signals:

    private:
      int*    currentProt;
      int*    currentLig;
      int     id;
      int     mode;
  };

  /***************************************/
  /******* WheeledValueLabel *************/
  /***************************************/
  
  class WheeledValueLabel : public QLabel
  {
    Q_OBJECT

    public:
      
      WheeledValueLabel(float * _value, float _minVal, float _maxVal, QWidget *parent = 0);
      
      void wheelEvent(QWheelEvent *event);
    
    signals:
      void wheelSignal();

    private:
      float*    value;
      float     minVal;
      float     maxVal;
  };
}

#endif