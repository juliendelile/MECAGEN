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

#include "customparamwidget.hpp"

#include "window.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>

namespace mg{

  //
  // Constructor/destructor:
  //

  CustomParamWidget::CustomParamWidget(Window *_win, Param<HOST> * _p, QWidget *parent): window(_win), param(_p)
  {


    setWindowFlags(Qt::WindowTitleHint | Qt::CustomizeWindowHint);

    QHBoxLayout * sliderLayout = new QHBoxLayout;

    setLayout(sliderLayout);
    
    setMinimumSize(300,300);

    setWindowTitle(tr("No Default Param"));

  }

  CustomParamWidget::~CustomParamWidget() { }

  //
  // Hints:
  //

  QSize CustomParamWidget::minimumSizeHint() const { return QSize(50, 50); }

  QSize CustomParamWidget::sizeHint() const { return QSize(100, 300); }

  void CustomParamWidget::initSliders()
  {
    
  }

}
