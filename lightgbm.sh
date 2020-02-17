#!/bin/bash

echo "LightGBM Model"

python -m main.train.main \
--model lightgbm \
--mode train \
--fold 4 \
--lr 0.05

python -m main.train.main \
--model lightgbm \
--mode test \
--fold 4
