#! /bin/bash -ex

export PYTHONPATH=../:$PYTHONPATH
scripts=../scripts

if [[ $1 != "" ]]; then
python $scripts/set_up_project.py --settings $1/settings.json
else
echo "Project not selected!"
fi
