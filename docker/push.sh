mkdir docker
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/dockerfiles/bashrc
docker build -t tilayealemu/melanet:latest .
docker push tilayealemu/melanet:latest
