#!/bin/bash

dataset_values=(377 386 387 392 393 394)
cuda_values=(0 1 2 3 4 5 6 7)

contains_element () {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

if ! [[ "$1" =~ ^[0-9]+$ ]] || ! contains_element "$1" "${dataset_values[@]}"; then
  echo "Error: Argument 1 is invalid. It must be one of ${dataset_values[*]}."
  exit 1
fi

if ! [[ "$2" =~ ^[0-9]+$ ]] || ! contains_element "$2" "${cuda_values[@]}"; then
  echo "Error: Argument 2 is invalid. It must be one of ${cuda_values[*]}."
  exit 1
fi

echo "Training on dataset zjumocap_${1}_mono with CUDA_VISIBLE_DEVICES=$2"
echo "Training on dataset zjumocap_${1}_mono with CUDA_VISIBLE_DEVICES=$2" >> exp/${1}.log
export CUDA_VISIBLE_DEVICES=${2}
python train.py dataset=zjumocap_${1}_mono >> exp/${1}.log 2>>exp/${1}_2.log
python render.py mode=test dataset.test_mode=view dataset=zjumocap_${1}_mono >> exp/${1}.log 2>>exp/${1}_2.log
