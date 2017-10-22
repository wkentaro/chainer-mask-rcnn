#!/bin/bash

set -x

pip install -q pytest
pytest -vs tests

set +x
