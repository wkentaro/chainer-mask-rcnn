#!/bin/bash -x

if [ "$(hostname)" != "hoop" ]; then
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $HERE

for server in dlboxs1 dlbox1 dlbox3 dlbox4 dlbox5 crux; do
  timeout 1 ssh $server ls &>/dev/null &&
    rsync -avt $server:chainer-mask-rcnn/experiments/mask_rcnn/logs/ logs/ &
done
wait
