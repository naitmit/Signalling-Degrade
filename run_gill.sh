#!/bin/bash

#-------------COMMENTS-------------
#runs script
#----------------------------------

# A bash script to run the python code

echo "Running Gillespie script"
time python3.7 trajectory_analysis.py

day=$(date +'%m'-'%d') #date in mm-dd format
now=$(date +'%R') #time in 24hr clock

# if [ -e ./plots ]
# then
#     echo "Removing previous snapshots"
#     rm -r ./snapshots
# fi

#option for storing frames folder in timestamped repository
if [ ! -e ./Gillespie ]
then
    echo "Making Gillespie folder"
    mkdir ./Gillespie
fi

if [ ! -e ./Gillespie/$day ]
then
    echo "Making day folder"
    mkdir ./Gillespie/$day
fi

echo "Making time folder"
mkdir ./Gillespie/$day/$now


echo "Moving files"
mv gill_var* ./output/* ./Gillespie/$day/$now
echo "Done"

exit
