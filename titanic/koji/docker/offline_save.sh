docker build -t ml -f DockerFile .
docker save ml > ml_3_6_img
echo 'saving docker image now...'
tar zcvf ml_3_6_img.tar.gz ml_3_6_img
