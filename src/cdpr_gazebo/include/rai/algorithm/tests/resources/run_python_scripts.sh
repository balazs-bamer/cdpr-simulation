#!/bin/bash

source ~/.virtualenvs/tensorflow/bin/activate

rm *.pb

python generate_policy.py
python generate_Qfunction.py