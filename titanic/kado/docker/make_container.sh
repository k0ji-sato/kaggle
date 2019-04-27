cd ../
docker build -t ml -f docker/DockerFile ./docker/
docker run -it -v $(pwd):/workdir -w /workdir --name machine_learning ml:3.6 /bin/bash

#when --rm is commanded, container will not be made
#docker run --rm -it -v $(pwd):/workdir -w /workdir ml:3.6 /bin/bash
