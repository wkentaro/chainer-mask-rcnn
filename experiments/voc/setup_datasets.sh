#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd $HERE

if [ "$(hostname)" != "hoop" ]; then
  exit 1
fi

set -x
set -e

for server in dlbox1 dlbox2 dlbox3 dlbox4 dlbox5 dlbox6 dlbox7 dlbox8 dlbox9 dlboxs1; do
  timeout 1 ssh $server ls &>/dev/null || continue
  rsync -avt --progress ~/data/datasets/pfnet/ $server:data/datasets/pfnet/ &
  rsync -avt --progress ~/data/datasets/VOC/ $server:data/datasets/VOC/ &
  rsync -avt --progress ~/data/datasets/COCO/ $server:data/datasets/COCO/ &
done
wait

set +x
set +e
