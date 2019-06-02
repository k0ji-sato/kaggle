#!/bin/bash

imagename="ml:3.6"
containername="MLforkaggle"

cd ./..
docker build -t $imagename -f docker/DockerFile ./docker 
docker run -it -v $(pwd):/workdir -w /workdir --name $containername -p 8888:8888 $imagename /bin/bash

