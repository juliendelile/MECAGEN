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

#include "qtobjects.hpp"

namespace mg{

  ColorLabel::ColorLabel(float* _colorR, float* _colorG, float* _colorB, QWidget *parent)
      : QLabel(parent), colorR(_colorR), colorG(_colorG), colorB(_colorB)
  {}

  void ColorLabel::mouseDoubleClickEvent ( QMouseEvent * event )
  {
      QColor qcolor = QColorDialog::getColor( QColor(255.0 * *colorR, 255.0 * *colorG,255.0 * *colorB ) , this);
      
      if (qcolor.isValid()){ 
        *colorR = (double)qcolor.red() / 255.0;
        *colorG = (double)qcolor.green() / 255.0;
        *colorB = (double)qcolor.blue() / 255.0;
        
        char string[300];
        sprintf(string, "QLabel {background-color: rgb(%d,%d,%d);}",
              (int)(255.0* *colorR),(int)(255.0* *colorG),(int)(255.0* *colorB) );
        this->setStyleSheet(string);
      }
      
      emit doubleClicked();
  }


  ProtLigRadioButton::ProtLigRadioButton(int* _currentProt, int* _currentLig, int _id, int _mode, const QString & text, QWidget *parent)
    : QRadioButton(text, parent), currentProt(_currentProt), currentLig(_currentLig), id(_id), mode(_mode)
  {
    setFixedWidth(150);
  }
  
  void ProtLigRadioButton::clicked(){
    if(mode == 0){
      *currentProt = id;
      *currentLig = -1;
    }
    else if(mode == 1){
      *currentProt = -1;
      *currentLig = id;
    }
    else if(mode == -1){
      *currentProt = -1;
      *currentLig = -1;
    }

  }


  /***************************************/
  /******* WheeledValueLabel *************/
  /***************************************/

  WheeledValueLabel::WheeledValueLabel(float * _value, float _minVal, float _maxVal, QWidget *parent)
      : QLabel(parent), value(_value), minVal(_minVal), maxVal(_maxVal)
  {
    // QFont font;
    // font.setPointSize(32);
    // font.setBold(true);
    // setFont(font);
    
    setFrameStyle(QFrame::Box | QFrame::Plain);
    setLineWidth(2);
    
    setAlignment( Qt::AlignCenter );

    setText(QString("%1").arg(QString("%1").arg(((float)*value) , 0, 'f', 2 )));
  }

  void WheeledValueLabel::wheelEvent(QWheelEvent *event)
  {
    
    float currentVal = *value;
    uint numDigits = 
        (currentVal>=100000.0) ? 6 :(currentVal>=10000.0 ? 5 :(currentVal>=1000.0 ? 4 :( currentVal>=100.0 ? 3 : ( currentVal>=10.0 ? 2 : 1 ) ) ) ) ;
    float update = .05 * pow(10,numDigits-1);
    if(event->delta() < 0){
      update *= -1;
    }
    
    float newVal = currentVal + update;
    
    if(newVal  < minVal ){
      newVal = minVal;
    }
    if( newVal > maxVal ){
      newVal = maxVal;
    }
    
    setText(QString("%1").arg(QString("%1").arg( newVal, 0, 'f', 2 )));

    *value = newVal;
   
    emit wheelSignal();

    event->accept();

  }


}