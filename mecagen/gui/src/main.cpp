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

#include "cells_qapplication.hpp"

int main(int argc, char *argv[])
{
	mg::CellsQApplication * app = new mg::CellsQApplication(argc, argv);
	
	if( app->is_init_ok() == true){
	 	app->connect( app, SIGNAL( lastWindowClosed() ), app, SLOT( quit() ) );
		app->exec();
		delete app;
	}
	return 0;
}
