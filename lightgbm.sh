#!/bin/bash

do
	echo "LightGBM Model"

	python -m main.train.main \
	--model lighgbm \
	--mode train \

	python -m main.train.main \
	--model lightgbm \
	--mode test
done
