#!/bin/bash

# set -x

# check torch
python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Torch is not installed"
    exit 1
fi

# check python include
python3 -c "import sysconfig; print(sysconfig.get_path('include'))" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Python include is not installed"
    exit 1
fi

# check python lib
python3 -c "import sysconfig; print(sysconfig.get_path('stdlib'))" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Python lib is not installed"
    exit 1
fi

# check python version
python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' &> /dev/null
if [ $? -ne 0 ]; then
    echo "Python version is not detected"
    exit 1
fi

# set +x
echo "Check build env success"
