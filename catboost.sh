#!/bin/bash

do
	echo "CatBoost Model"

	python -m main.train.main \
	--model catboost \
	--mode train \

	python -m main.train.main \
	--model catboost \
	--mode test
done
