#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -x

cd $HERE

if [ "$(hostname)" != "hoop" ]; then
  exit 1
fi

for server in hoop green dlbox1 dlbox2 dlbox3 dlbox4 dlbox5 baxter-c1; do
  [ "$server" = "hoop" ] && continue
  timeout 1 ssh $server ls &>/dev/null || continue
  rsync -avt $server:mask-rcnn/experiments/mask_rcnn/logs/ logs/
done

set +x
