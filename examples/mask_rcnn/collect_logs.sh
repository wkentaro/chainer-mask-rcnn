#!/bin/bash

set -x

for server in hoop green dlbox1 dlbox2 dlbox3 dlbox4 dlbox5; do
  echo $server
  rsync -avt $server:mask-rcnn/experiments/mask_rcnn/logs .
done

set +x
