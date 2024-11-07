#!/bin/bash
set -m

maxProcs=4
declare -A currentJobs=( )

a=4
n=80000
s=500
timesteps=50000

for idx in 0 1 2 3 4 5 6 7 8 9
do
    for noise in 0.01 0.02
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
