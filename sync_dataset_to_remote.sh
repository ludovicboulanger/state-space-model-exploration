#!/bin/bash
if [ "$#" -ne 5 ]; then
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

shift $((OPTIND - 1))
dataset_path=$1

rsync -P -avz -e "ssh -i $keyfile" $dataset_path $username@$remotename:$destination
