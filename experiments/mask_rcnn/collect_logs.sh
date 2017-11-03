#!/bin/bash -x

if [ "$(hostname)" != "hoop" ]; then
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $HERE

for server in green dlbox1 dlbox2 dlbox3 dlbox4 dlbox5 baxter-c1; do
  timeout 1 ssh $server ls &>/dev/null &&
    rsync -avt $server:mask-rcnn/experiments/mask_rcnn/logs/ logs/ &
done
wait
