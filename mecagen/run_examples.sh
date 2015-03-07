#!/bin/bash

echo "Please select one of the following examples:"

PS3=''

option1="Signal Mediated-Toggle Switch             (MECAGEN must be compiled in *default* mode)"
option2="Boundary Formation and Epithelialization  (MECAGEN must be compiled in *default* mode)"
option3="Zebrafish Epiboly Phase 1                 (MECAGEN must be compiled in *zebrafish* mode)"

options=("$option1" "$option2" "$option3" "Quit")

select opt in "${options[@]}"
do
    case $opt in
        "$option1")
            ./launchGUI.sh -param examples/default/SignalMedToggleSwitch_FlatTissue_grn_default/param/param_archive.xml -metaparam examples/default/SignalMedToggleSwitch_FlatTissue_grn_default/state/metaparam_archive.xml -state examples/default/SignalMedToggleSwitch_FlatTissue_grn_default/state/state_archive.xml
            break
            ;;
        "$option2")
            ./launchGUI.sh -param examples/default/Boundaryformation_epithelialization/param/param_archive.xml -metaparam examples/default/Boundaryformation_epithelialization/state/metaparam_archive.xml -state examples/default/Boundaryformation_epithelialization/state/state_archive.xml
            break
            ;;
        "$option3")
            ./launchGUI.sh -param examples/zebrafish/Zebrafish_epiboly/param_archive.xml -metaparam examples/zebrafish/Zebrafish_epiboly/metaparam_archive.xml -state examples/zebrafish/Zebrafish_epiboly/state_archive.xml
            break
            ;;
        "Quit")
            break
            ;;
        *) echo invalid option;;
    esac
done
