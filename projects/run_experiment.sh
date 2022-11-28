#! /bin/bash

export PYTHONPATH=../:$PYTHONPATH
scripts=../scripts

# Check selected experiment
if [[ $1 == "" ]]; then
    echo "Experiment not selected!"
    exit 1
fi

# Check selected config file
if [[ $2 == "" ]]; then
    echo "Config file not selected!"
    exit 1
fi

# Check cache option
if [[ $3 == "-c" ]]; then
    find $1/results/$2 -maxdepth 1 ! -name 'cache' -not -path "$1/results/$2" -exec rm -rf {} +
elif [[ $3 == "" ]]; then
    rm -rf $1/results/$2/
else
    echo "Only cache flag is available"
    exit 1
fi

# Run experiment
python $scripts/run_pipeline.py --config $1/configs/$2.py
