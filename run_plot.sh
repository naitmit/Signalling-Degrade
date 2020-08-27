#!/bin/bash

#-------------COMMENTS-------------
#runs script
#----------------------------------

# A bash script to run the python code
for i in {1..10} #create directories for programs to run in
do
  if [ ! -e ./plot_progress${i} ]
  then
      mkdir ./plot_progress${i}
      echo "New directory made: plot_progress"${i}

      cp steady_deg_accuracy_plots_run* laguerre_testing* params* formulae* settings* variables* ./plot_progress${i}
      echo "Staged Python files in plot_progress"${i}

      cd ./plot_progress${i}
      echo "Working directory is now plot_progress"${i}

      echo "Running plotting script"
      time python3.7 steady_deg_accuracy_plots_run.py

      rm steady_deg_accuracy_plots_run* laguerre_testing* params* formulae* settings*

      day=$(date +'%m'-'%d') #date in mm-dd format
      now=$(date +'%R') #time in 24hr clock

      #create directory to store files in
      if [ ! -e ../plotting ]
      then
          echo "Making plotting folder"
          mkdir ./plotting
      fi

      if [ ! -e ../plotting/$day ]
      then
          echo "Making day folder"
          mkdir ../plotting/$day
      fi

      if [ ! -e ../plotting/$day/$now ] #if program finishes on a different minute
      then
          cd ../
          echo "Moving files"
          mv ./plot_progress${i} ./plotting/$day

          echo "Renaming files to "${now}
          cd ./plotting/$day
          mv ./plot_progress${i} ./$now
          echo "Done"
      else #minute folder already exists
          echo "Time folder already exists"
          cd ../
          mv ./plot_progress${i} ./plotting/$day/$now #move into folder
          cd ./plotting/$day/$now
          echo "bro"

          for j in {1..5} #check names
          do
            echo $j
            if [ ! -e ./plot${j} ]
            then
            mv ./plot_progress${i} ./plot${j} #rename it
            break
            fi
          done
      fi
      break
  fi
done
