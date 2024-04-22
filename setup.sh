#!/bin/sh


workingDirectory=$(pwd)
echo $workingDirectory
export PYTHONPATH=$workingDirectory
export PYTHONPATH=$PYTHONPATH:$workingDirectory