#!/bin/bash

imagename="python"
containername="python_exampe"

cd ./..
docker run -it --name $containername -p 8888:8888 $imagename /bin/bash

