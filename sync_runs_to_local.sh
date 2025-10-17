#!/bin/bash
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 [k]eyfile [u]sername [r]emote [s]ource [d]estination"
  exit 1
fi

while getopts k:u:r:s:d: flag; do
  case "${flag}" in
  k) keyfile=${OPTARG} ;;
  u) username=${OPTARG} ;;
  r) remotename=${OPTARG} ;;
  s) source=${OPTARG} ;;
  d) destination=${OPTARG} ;;
  esac
done

rsync -avz -e "ssh -i $keyfile" $username@$remotename.alliancecan.ca:/home/$username/projects/def-seanwood/$username/training-runs/$source $destination
