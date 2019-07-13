#!/bin/bash

export PYTHONPATH="../:${PYTHONPATH}"

python plot_profile.py $@
