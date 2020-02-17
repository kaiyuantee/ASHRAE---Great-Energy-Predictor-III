#!/bin/bash


echo "ASHRAE - Great Energy Predictor III"

sh ./directories.sh
sh ./lightgbm.sh
sh ./catboost.sh
sh ./keras,sh
sh ./ensemble.sh



