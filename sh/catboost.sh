#!/bin/bash

echo "CatBoost Model"

python -m main.train.main \
--model catboost \
--mode train \
--fold 4 \
--lr 0.1

python -m main.train.main \
--model catboost \
--mode test \
--fold 4

