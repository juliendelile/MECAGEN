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

#include "recordingwidget.hpp"

// #include <stdlib.h>
#include <QVBoxLayout>
#include <QString>
#include <QFileDialog>
#include <QCoreApplication>

namespace mg{

  //
  // Constructor/destructor:
  //

  RecordingWidget::RecordingWidget(GLWidget *_glw, QWidget *parent): glwidget(_glw), imageAdress("/tmp")
  {

    setWindowFlags(Qt::WindowTitleHint | Qt::CustomizeWindowHint);

    startStopRecordButton = new QPushButton("Start Image Sequence", this);
    connect(startStopRecordButton,  SIGNAL(pressed()), this, SLOT(startStopRecording()));

    snapshotButton = new QPushButton("Take Snapshot", this);
    connect(snapshotButton, SIGNAL(pressed()), this, SLOT(takeSingleSnapshot()));

    imageAdressButton = new QPushButton(imageAdress, this);
    connect(imageAdressButton,  SIGNAL(pressed()), this, SLOT(updateImageAdress()));

    QVBoxLayout * vlayout = new QVBoxLayout;
    vlayout->addWidget(snapshotButton);
    vlayout->addWidget(startStopRecordButton);
    vlayout->addWidget(imageAdressButton);
    
    setLayout(vlayout);
    
    setMinimumSize(300,300);

    setWindowTitle(tr("Display Control"));

  }

  RecordingWidget::~RecordingWidget() { }

  //
  // Hints:
  //

  QSize RecordingWidget::minimumSizeHint() const { return QSize(50, 50); }

  QSize RecordingWidget::sizeHint() const { return QSize(100, 300); }

  void RecordingWidget::startStopRecording(){
    if(glwidget->recording==true){
      glwidget->recording = false;
      glwidget->unlockConsumer();
      startStopRecordButton->setText("Start recording");
    }
    else{
      glwidget->recording_timer = 0;
      glwidget->recording_period=2;
      glwidget->recording_folder = imageAdress;
      glwidget->recording=true;
      startStopRecordButton->setText("Stop recording");
    }
  }  

  void RecordingWidget::takeSingleSnapshot(){
    glwidget->recording_folder = imageAdress;
    glwidget->record_single_snapshot=true;
  }

  void RecordingWidget::updateImageAdress(){

    imageAdress = QFileDialog::getExistingDirectory(
                              this, 
                              tr("Select/create directory"), 
                              imageAdress,
                              QFileDialog::ShowDirsOnly
                              );
 
    imageAdressButton->setText(imageAdress);
  }

}
