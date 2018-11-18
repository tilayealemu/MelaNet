mkdir docker
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/dockerfiles/bashrc
docker build -t tilayealemu/quanquanet:latest .
docker push tilayealemu/quanquanet:latest
