#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -x

cd $HERE

if [ "$(hostname)" != "hoop" ]; then
  exit 1
fi

for server in dlboxs1 dlbox1 dlbox3 dlbox4 dlbox5 crux; do
  timeout 1 ssh $server ls &>/dev/null || continue
  rsync -avt --progress ~/data/datasets/pfnet/ $server:data/datasets/pfnet/
  rsync -avt --progress ~/data/datasets/VOC/ $server:data/datasets/VOC/
  rsync -avt --progress ~/data/datasets/COCO/ $server:data/datasets/COCO/
done

set +x
