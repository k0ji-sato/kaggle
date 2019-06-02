echo 'reading docker image now...'
tar zxvf ml_3_6_img.tar.gz
docker load < ml_3_6_img
cd ../
docker run -it -v $(pwd):/workdir -w /workdir ml:3.6 /bin/bash
