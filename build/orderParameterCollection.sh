#!/bin/bash
set -m

maxProcs=12
declare -A currentJobs=( )

a=4
n=80000
s=500
timesteps=30000

for idx in 0 1 2 3 4
do
    for noise in 0.02 0.03 0.035 0.04 0.045 0.05 0.055 0.06 0.07 0.08 0.09 0.1
    do
        #if we are over our limit of concurrent jobs, wait for one to finish
        if (( ${#currentJobs[@]} >= maxProcs )); then
            wait -p finishedJob -n
            # we just grabbed the id of the job that finished; remove it from our array
            unset currentJobs[$finishedJob]
        fi
        nohup ./timeOrderedXY.out "-a" $a "-e" $noise "-n" $n "-t" $timesteps "-s" $s "-x" $idx &> "scriptOutput${idx}_${noise}.txt" &
       currentPID=$!
    done
done
wait
# Output to screen when this finished... should be 40 seconds
currentTime="$(date +"%T")"
echo "jobs finished ${currentTime}"
