# wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/dockerfiles/bashrc

docker login -u=tilayealemu -p=$DOCKERHUB_PASS
docker build -t tilayealemu/melanet:latest .
docker push tilayealemu/melanet:latest