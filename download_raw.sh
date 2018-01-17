#! /bin/bash
rm data -rf
orig_url="https://physionet.org/challenge/2017/training2017.zip"
label_fix_url="https://physionet.org/challenge/2017/REFERENCE-v3.csv"
wget -nc $orig_url -O "data.zip"
unzip -nq "data.zip"  && mv "training2017" "data" && cd "data"
rm "REFERENCE.csv"
wget -nc $label_fix_url -O "REFERENCE.csv"
