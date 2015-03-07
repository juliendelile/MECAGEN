#!/bin/bash

# This is used to manage backends: recognize the words "cpp", "omp" and "cuda"
declare -A array
for constant in "cpp" "omp" "cuda"
do
  array[$constant]=1
done

# This is used to find the actual location of the launching script.
# It is important as our ressources (.so and executable) have paths relative to that location.
SCRIPT_ROOT="${BASH_SOURCE[0]}"
# Read symlink while we do not have the fully resolved path
# man test: -h: FILE exists and is a symbolic link
while [ -h "$SCRIPT_ROOT" ]; do
  DIR="$( cd -P "$( dirname "$SCRIPT_ROOT" )" && pwd )"
  SCRIPT_ROOT="$(readlink "$SCRIPT_ROOT")"
  # If we have a relative symlink (i.e. NOT an absolute path starting with '/'), construct the full path
  [[ $SCRIPT_ROOT != /* ]] && SCRIPT_ROOT="$DIR/$SCRIPT_ROOT"
done
DIR="$( cd -P "$( dirname "$SCRIPT_ROOT" )" && pwd )"

# Build LD_LIBRARY_PATH according to DIR, COMPONENTS and the backend provide as argument
COMPONENTS=('model' 'producers' 'consumers')
build_ld_path()
{
  path=""
  for item in ${COMPONENTS[@]}; do
    path="$path:$DIR/$item/lib/$1"
  done
  echo $path
}

# Now manage the LD_LIBRARY_PATH according to the args
if [[ -z "$1" ]]
then
  # Arg1 is not defined, use cpp by default:
  export LD_LIBRARY_PATH=$(build_ld_path cpp)":$LD_LIBRARY_PATH"
else
  if [[ ${array["$1"]} ]];
  then
    # Arg is ok:
  export LD_LIBRARY_PATH=$(build_ld_path $1)":$LD_LIBRARY_PATH"
    shift # shift all args on the left $1 $2 $3 => $2 $3
  else
    # Arg is not ok, use cpp by default:
  export LD_LIBRARY_PATH=$(build_ld_path cpp)":$LD_LIBRARY_PATH"
  fi
fi

$DIR/gui/bin/mecagenGUI "$@"
# valgrind --leak-check=yes $DIR/gui/bin/mecagenGUI "$@"
# gdb -ex=r --args $DIR/gui/bin/mecagenGUI "$@"