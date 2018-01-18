#!/bin/bash
#SBATCH --gres=gpu:k80:4
#SBATCH --ntasks-per-node=4
scripts='python3 main.py --debug -a vgg16_bn' 'python3 main.py --debug -a vgg19_bn'
for i in $(scripts) 
do
  echo "$i"
  #srun "$i" &
done

