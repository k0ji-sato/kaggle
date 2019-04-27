cd ../
docker build -t ml:3.6 -f docker/DockerFile ./docker/
docker run -it -v $(pwd):/workdir -w /workdir --name machine_learning -p 8000:8000 ml:3.6 /bin/bash

#when --rm is commanded, container will not be made
#docker run --rm -it -v $(pwd):/workdir -w /workdir ml:3.6 /bin/bash
