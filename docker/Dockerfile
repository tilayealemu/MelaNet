# Unless stated, all taken from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/cpu-jupyter.Dockerfile

FROM ubuntu:16.04

ARG USE_PYTHON_3_NOT_2=True
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} install --upgrade \
    pip \
    setuptools

ARG TF_PACKAGE=tensorflow
RUN ${PIP} install ${TF_PACKAGE}

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

RUN ${PIP} install jupyter

RUN mkdir /notebooks && chmod a+rwx /notebooks
RUN mkdir /.local && chmod a+rwx /.local
WORKDIR /notebooks
EXPOSE 8888

# Start https://github.com/udacity/AIND-VUI-Capstone
RUN pip install keras librosa python-speech-features soundfile==0.9.0.post1 matplotlib pydot
# RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /install-miniconda.sh && /bin/bash /install-miniconda.sh -b -p /opt/miniconda
RUN apt-get install wget libsndfile1 libav-tools vim graphviz -y
# End https://github.com/udacity/AIND-VUI-Capstone

#Optional: Support AVX AVX2 https://stackoverflow.com/questions/47068709
RUN pip install --ignore-installed --upgrade "https://github.com/lakshayg/tensorflow-build/releases/download/tf1.11.0-ubuntu16.04-py2.7-py3.5/tensorflow-1.11.0-cp35-cp35m-linux_x86_64.whl" --user

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/notebooks --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token=''"]