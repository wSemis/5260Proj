#!/bin/bash

# Define an array of numbers
numbers=(386 387 393)
mkdir -p exp
echo $1 >> exp/info.txt
# Check if a tmux session exists, else create one
session="training_session"
tmux has-session -t $session 2>/dev/null

if [ $? != 0 ]; then
    tmux new-session -d -s $session
fi

# Loop through the array with indices and create a new window for each command
for i in "${!numbers[@]}"; do
    exp_num=${numbers[$i]}
    echo "Setting up tmux window for num=$exp_num and cuda=$i"
    # Create a new window in tmux session and run the script
    tmux new-window -t $session -n "train_$exp_num" "./train.sh $exp_num $i"
done

tmux detach -s $session

