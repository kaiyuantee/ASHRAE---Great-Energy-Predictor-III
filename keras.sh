#!/bin/bash

echo "Keras Embedding Model"

python -m main.train.main \
--model keras \
--mode train \
--fold 4 \
--batch_size 1024 \
--lr 0.001 \
--epochs 10 \

python -m main.train.main \
--model keras \
--mode test \
--fold 4 \
--batch_size 5000 \


