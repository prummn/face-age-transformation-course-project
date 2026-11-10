#!/bin/bash
cd -- "$(dirname -- "$0")" || exit 1
python3 integration/launch.py
if [ $? -ne 0 ]; then
  echo "请保留上方错误信息。按回车关闭。"
  read -r
fi
