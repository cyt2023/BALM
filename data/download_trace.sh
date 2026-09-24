#!/bin/sh
# 下载 Azure Functions 2019 trace (CC-BY 4.0)。
# 引用：M. Shahrad et al., "Serverless in the Wild: Characterizing and Optimizing the
#       Serverless Workload at a Large Cloud Provider", USENIX ATC 2020.
set -e
DEST="${1:-/tmp/aztrace}"
mkdir -p "$DEST/data"
cd "$DEST"
if [ ! -f az2019.tar.xz ]; then
  curl -L -o az2019.tar.xz \
    "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz"
fi
tar -xJf az2019.tar.xz -C data
echo "解压完成: $DEST/data"
echo "运行构建脚本: AZ_TRACE_DIR=$DEST/data python3 code/00_build_dataset.py"
