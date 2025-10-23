#!/bin/bash
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 [k]eyfile [u]sername [r]emotename [d]estination"
  exit 1
fi

while getopts k:u:r:d: flag; do
  case "${flag}" in
  k) keyfile=${OPTARG} ;;
  u) username=${OPTARG} ;;
  r) remotename=${OPTARG}.alliancecan.ca ;;
  d) destination=${OPTARG} ;;
  esac
done

SCRIPT_PATH="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"

rsync \
  --exclude "__pycache__" \
  --exclude ".*" \
  --exclude "data" \
  --exclude "training-runs" \
  -avz -e "ssh -i $keyfile" $SCRIPT_PATH $username@$remotename:$destination
