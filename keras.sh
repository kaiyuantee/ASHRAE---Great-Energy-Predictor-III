#!/bin/bash

do
	echo "Keras Embedding Model"

	python -m main.train.main \
	--model keras \
	--mode train \

	python -m main.train.main \
	--model keras \
	--mode test
done
