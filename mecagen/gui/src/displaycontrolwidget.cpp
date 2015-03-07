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

#include "displaycontrolwidget.hpp"

// #include <stdlib.h>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QButtonGroup>
#include <QRadioButton>
#include <QCheckBox>
#include <QGroupBox>
#include <QComboBox>

#include "qtobjects.hpp"

namespace mg{

  //
  // Constructor/destructor:
  //

  DisplayControlWidget::DisplayControlWidget(GLWidget *_glw, QWidget *parent, Param<HOST> *_p): glwidget(_glw), param(_p)
  {

    setWindowFlags(Qt::WindowTitleHint | Qt::CustomizeWindowHint);

    /**********************/
    /*****   Slicers  *****/
    /**********************/

    sliderSlicerXmax = new QSlider(Qt::Horizontal, this); sliderSlicerXmax->setRange(-100,100); sliderSlicerXmax->setSingleStep(.02);
    sliderSlicerYmax = new QSlider(Qt::Horizontal, this); sliderSlicerYmax->setRange(-100,100); sliderSlicerYmax->setSingleStep(.02);
    sliderSlicerZmax = new QSlider(Qt::Horizontal, this); sliderSlicerZmax->setRange(-100,100); sliderSlicerZmax->setSingleStep(.02);
    sliderSlicerXmin = new QSlider(Qt::Horizontal, this); sliderSlicerXmin->setRange(-100,100); sliderSlicerXmin->setSingleStep(.02);
    sliderSlicerYmin = new QSlider(Qt::Horizontal, this); sliderSlicerYmin->setRange(-100,100); sliderSlicerYmin->setSingleStep(.02);
    sliderSlicerZmin = new QSlider(Qt::Horizontal, this); sliderSlicerZmin->setRange(-100,100); sliderSlicerZmin->setSingleStep(.02);

    sliderSlicerXmax->setValue(glwidget->SlicerXmax);
    sliderSlicerXmin->setValue(glwidget->SlicerXmin);
    sliderSlicerYmax->setValue(glwidget->SlicerYmax);
    sliderSlicerYmin->setValue(glwidget->SlicerYmin);
    sliderSlicerZmax->setValue(glwidget->SlicerZmax);
    sliderSlicerZmin->setValue(glwidget->SlicerZmin);

    connect(sliderSlicerXmax, SIGNAL(valueChanged(int)), glwidget, SLOT(setSlicerXmax(int)));
    connect(sliderSlicerYmax, SIGNAL(valueChanged(int)), glwidget, SLOT(setSlicerYmax(int)));
    connect(sliderSlicerZmax, SIGNAL(valueChanged(int)), glwidget, SLOT(setSlicerZmax(int)));
    connect(sliderSlicerXmin, SIGNAL(valueChanged(int)), glwidget, SLOT(setSlicerXmin(int)));
    connect(sliderSlicerYmin, SIGNAL(valueChanged(int)), glwidget, SLOT(setSlicerYmin(int)));
    connect(sliderSlicerZmin, SIGNAL(valueChanged(int)), glwidget, SLOT(setSlicerZmin(int)));


    QVBoxLayout * vlayout = new QVBoxLayout;

    QGroupBox *slicerGroupBox = new QGroupBox(tr("Slicers"));
    QVBoxLayout * slicerVLay = new QVBoxLayout;

    QHBoxLayout * layxmax = new QHBoxLayout;
    layxmax->addWidget(new QLabel(tr("Xmax")));
    layxmax->addWidget(sliderSlicerXmax);
    slicerVLay->addLayout(layxmax);

    QHBoxLayout * layxmin = new QHBoxLayout;
    layxmin->addWidget(new QLabel(tr("Xmin")));
    layxmin->addWidget(sliderSlicerXmin);
    slicerVLay->addLayout(layxmin);

    QHBoxLayout * layymax = new QHBoxLayout;
    layymax->addWidget(new QLabel(tr("Ymax")));
    layymax->addWidget(sliderSlicerYmax);
    slicerVLay->addLayout(layymax);

    QHBoxLayout * layymin = new QHBoxLayout;
    layymin->addWidget(new QLabel(tr("Ymin")));
    layymin->addWidget(sliderSlicerYmin);
    slicerVLay->addLayout(layymin);

    QHBoxLayout * layzmax = new QHBoxLayout;
    layzmax->addWidget(new QLabel(tr("Zmax")));
    layzmax->addWidget(sliderSlicerZmax);
    slicerVLay->addLayout(layzmax);

    QHBoxLayout * layzmin = new QHBoxLayout;
    layzmin->addWidget(new QLabel(tr("Zmin")));
    layzmin->addWidget(sliderSlicerZmin);
    slicerVLay->addLayout(layzmin);
    
    slicerGroupBox->setLayout(slicerVLay);
    vlayout->addWidget(slicerGroupBox);

    /**********************/
    /*** Projection mode **/
    /**********************/

    QButtonGroup* projButtonGroup = new QButtonGroup;
    QRadioButton* button1 = new QRadioButton("Orthographic" , this);
    QRadioButton* button2 = new QRadioButton("Perspective" , this);
    projButtonGroup->addButton(button1);
    projButtonGroup->addButton(button2);

    if(glwidget->orthoProjection == true){
      button1->setChecked(true);
    }
    else{
      button2->setChecked(true);
    }

    QHBoxLayout * layproj = new QHBoxLayout;
    layproj->addWidget(button1);
    layproj->addWidget(button2);
    
    QGroupBox *projGroupBox = new QGroupBox(tr("Projection Mode"));
    projGroupBox->setLayout(layproj);

    vlayout->addWidget(projGroupBox);
    
    connect(button1, SIGNAL(toggled(bool)), glwidget, SLOT(setOrthoProj()));
    connect(button2, SIGNAL(toggled(bool)), glwidget, SLOT(setPerspProj()));

    /**********************/
    /**Proteins / Ligands**/
    /**********************/

    QGroupBox *colGroupBox = new QGroupBox(tr("Cell coloring"));
    QButtonGroup* protButtonGroup = new QButtonGroup;
    QVBoxLayout* colVLay = new QVBoxLayout;

    // QRadioButton* noneButton = new QRadioButton("none", this); 
    ProtLigRadioButton* noneButton = new ProtLigRadioButton(&(glwidget->displayParams.currentProtein), &(glwidget->displayParams.currentLigand), 0, -1, "none", this);
    connect(noneButton, SIGNAL(toggled(bool)), noneButton, SLOT(clicked()));
    noneButton->setChecked(true);
    QHBoxLayout* laynone = new QHBoxLayout;
    laynone->addWidget(noneButton); 
    protButtonGroup->addButton(noneButton); // for exclusion scope
    
    ColorLabel* noneLab = new ColorLabel(&(glwidget->displayParams.noneColorR),&(glwidget->displayParams.noneColorG),&(glwidget->displayParams.noneColorB));
    char string[300];
    sprintf(string, "QLabel {background-color: rgb(%d,%d,%d);}",
          (int)(255.0* glwidget->displayParams.noneColorR),(int)(255.0*glwidget->displayParams.noneColorG),(int)(255.0*glwidget->displayParams.noneColorB) );
    noneLab->setStyleSheet(string);
    laynone->addWidget(noneLab);
  
    colVLay->addLayout(laynone);

    #ifdef CUDA_VBO 
      connect(noneLab, SIGNAL(doubleClicked()),glwidget,SLOT(newDisplayParamVBO())); 
      connect(noneButton, SIGNAL(toggled(bool)), glwidget,SLOT(newDisplayParamVBO()));
    #endif

    // Color by protein
    for(uint i=0; i < param->numProteins[0]; i++ ){
      
      ProtLigRadioButton* protButton = new ProtLigRadioButton(
                    &(glwidget->displayParams.currentProtein), 
                    &(glwidget->displayParams.currentLigand), 
                    i, 0, param->proteins[i].name, this);
      connect(protButton, SIGNAL(toggled(bool)), protButton, SLOT(clicked()));
      QHBoxLayout* layprot = new QHBoxLayout;
      layprot->addWidget(protButton);
      protButtonGroup->addButton(protButton); // for exclusion scope

      ColorLabel* protLab = new ColorLabel(&(glwidget->displayParams.proteinColorR[i]),&(glwidget->displayParams.proteinColorG[i]),&(glwidget->displayParams.proteinColorB[i]));
      char string[300];
      sprintf(string, "QLabel {background-color: rgb(%d,%d,%d);}",
            (int)(255.0* glwidget->displayParams.proteinColorR[i]),(int)(255.0*glwidget->displayParams.proteinColorG[i]),(int)(255.0*glwidget->displayParams.proteinColorB[i]) );
      protLab->setStyleSheet(string);
      layprot->addWidget(protLab);

      WheeledValueLabel * protWheelLab = new WheeledValueLabel(
                                               &(glwidget->displayParams.proteinThreshold[i]), 
                                               .0, 10000.0, this);
      layprot->addWidget(protWheelLab);

      colVLay->addLayout(layprot);
      
      #ifdef CUDA_VBO 
        connect(protLab, SIGNAL(doubleClicked()),glwidget,SLOT(newDisplayParamVBO()));
        connect(protButton, SIGNAL(toggled(bool)), glwidget,SLOT(newDisplayParamVBO()));
        connect(protWheelLab, SIGNAL(wheelSignal()), glwidget,SLOT(newDisplayParamVBO()));
      #endif
    }

    for(uint i=0; i < param->numLigands[0]; i++ ){
      
      ProtLigRadioButton* ligButton = new ProtLigRadioButton(
                    &(glwidget->displayParams.currentProtein), 
                    &(glwidget->displayParams.currentLigand), 
                    i, 1, param->ligandParams[i].name, this);
      connect(ligButton, SIGNAL(toggled(bool)), ligButton, SLOT(clicked()));
      QHBoxLayout* laylig = new QHBoxLayout;
      laylig->addWidget(ligButton);
      protButtonGroup->addButton(ligButton); // for exclusion scope

      ColorLabel* ligLab = new ColorLabel(&(glwidget->displayParams.ligandColorR[i]),&(glwidget->displayParams.ligandColorG[i]),&(glwidget->displayParams.ligandColorB[i]));
      char string[300];
      sprintf(string, "QLabel {background-color: rgb(%d,%d,%d);}",
            (int)(255.0* glwidget->displayParams.ligandColorR[i]),(int)(255.0*glwidget->displayParams.ligandColorG[i]),(int)(255.0*glwidget->displayParams.ligandColorB[i]) );
      ligLab->setStyleSheet(string);
      laylig->addWidget(ligLab);

      WheeledValueLabel * ligWheelLab = new WheeledValueLabel(
                                              &(glwidget->displayParams.ligandThreshold[i]), 
                                              .0, 10000.0, this);
      laylig->addWidget(ligWheelLab);

      colVLay->addLayout(laylig);

      #ifdef CUDA_VBO 
        connect(ligLab, SIGNAL(doubleClicked()),glwidget,SLOT(newDisplayParamVBO()));
        connect(ligButton, SIGNAL(toggled(bool)), glwidget,SLOT(newDisplayParamVBO()));
        connect(ligWheelLab, SIGNAL(wheelSignal()), glwidget,SLOT(newDisplayParamVBO()));
      #endif
    }

    colGroupBox->setLayout(colVLay);
    vlayout->addWidget(colGroupBox);

    /**********************/
    /***      2D info    **/
    /**********************/
    
    QGroupBox *twoDGroupBox = new QGroupBox(tr("2D info"));
    twoDGroupBox->setCheckable(true);

    if(glwidget->twoDinfo == true){
      twoDGroupBox->setChecked(true);
    }
    else{
      twoDGroupBox->setChecked(false);
    }

    connect(twoDGroupBox, SIGNAL(toggled(bool)), glwidget, SLOT(set2Dinfo(bool)));

    // vlayout->addWidget(twoDGroupBox);

    /**********************/
    /***      Type       **/
    /**********************/

    QGroupBox *typeColGroupBox = new QGroupBox(tr("Type"));
    typeColGroupBox->setCheckable(true);
    if(glwidget->displayParams.colorByType == 0){
      typeColGroupBox->setChecked(false);
    }
    else{
      typeColGroupBox->setChecked(true);
    }
    connect(typeColGroupBox, SIGNAL(toggled(bool)), glwidget, SLOT(setColByType(bool)));
    // vlayout->addWidget(typeColGroupBox);

    /**********************/
    /***      Axes       **/
    /**********************/
    QGroupBox *axesGroupBox = new QGroupBox(tr("Axes"));
    axesGroupBox->setCheckable(true);
    if(glwidget->displayParams.drawAxes == 0){
      axesGroupBox->setChecked(false);
    }
    else{
      axesGroupBox->setChecked(true);
    }
    connect(axesGroupBox, SIGNAL(toggled(bool)), glwidget, SLOT(setDrawAxes(bool)));

    QComboBox* axesQCB = new QComboBox();
    for(int i=0; i<param->numPolarizationAxes[0]; i++){
      QString axename = QString("CandAxe_%1").arg(i);
      axesQCB->addItem(axename);
    }
    axesQCB->addItem("ABaxes");
    axesQCB->addItem("Axe1");
    connect(axesQCB, SIGNAL(activated(int)), glwidget, SLOT(setAxesId(int)));

    QHBoxLayout * hblay = new QHBoxLayout;
    hblay->addWidget(twoDGroupBox);
    hblay->addWidget(typeColGroupBox);
    hblay->addWidget(axesGroupBox);
    hblay->addWidget(axesQCB);

    vlayout->addLayout(hblay);

    /**********************/
    /***** Finish  ********/
    /**********************/
    setLayout(vlayout);
    
    setMinimumSize(350,700);

    setWindowTitle(tr("Display Control"));

  }

  DisplayControlWidget::~DisplayControlWidget() { }

  //
  // Hints:
  //

  QSize DisplayControlWidget::minimumSizeHint() const { return QSize(50, 50); }

  QSize DisplayControlWidget::sizeHint() const { return QSize(100, 300); }

}
