#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -x

cd $HERE

for server in hoop green dlbox1 dlbox2 dlbox3 dlbox4 dlbox5; do
  rsync -avt $server:mask-rcnn/experiments/mask_rcnn/logs/ logs/
done

set +x
