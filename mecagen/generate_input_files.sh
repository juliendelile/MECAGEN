#!/bin/bash

if [ -z "$1" ]
  then
    echo "No argument supplied. Please specify whether you want to generate \"all\" (./generate_input_files all) or \"default\" (./generate_input_files default) or \"zebrafish\" (./generate_input_files zebrafish) input files."
    exit
fi

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ok="ko"

if [ $1 = "default" -o $1 = "all" ]; then
	
	cd $DIR/examples/default/Boundaryformation_epithelialization/param
	make run
	cd ../state
	make run
	printf "\n*** The \"Boundary Formation and Epithelialization\" input files have been generated. ***\n\n"

	cd $DIR/examples/default/SignalMedToggleSwitch_FlatTissue_grn_default/param
	make run
	cd ../state
	make run
	printf "\n*** The \"Signal-Mediated Toggle Switch\" input files have been generated. ***\n\n"

	ok="ok"
fi

if [ $1 = "zebrafish" -o $1 = "all" ]; then
	
	cd $DIR/examples/zebrafish/Zebrafish_epiboly
	make run
	printf "\n*** The \"Zebrafish Epiboly\" input files have been generated. ***\n\n"

	ok="ok"
fi

if [ $ok = "ko" ]; then	
	echo "Wrong argument supplied. Please specify whether you want to generate \"all\" (./generate_input_files all) or \"default\" (./generate_input_files default) or \"zebrafish\" (./generate_input_files zebrafish) examples."
fi
