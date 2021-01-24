# custom miniconda build that contains only libraries that we really need
# we can specify for example numpy, pandas, matplotlib ...
# maintainer: tcoil.info
#
# build as
# sudo docker build -t custom_miniconda .
#
# run this image as
# coil@coil:~/Desktop/miniconda_docker_build$ sudo docker run --name custom_miniconda -i -t -p 8501:8501 custom_miniconda
# or with docker compose demonized


FROM centos:8


## Step 1:
# Create a working directory
WORKDIR /app


## Step 2:
# Copy source code to working directory
COPY . app.py /app/
COPY . iris-pytorch.pkl /app/

## Step 3:
# Install packages from requirements.txt
# it sees requirements.txt file in build dir
# no need to copy to docker image
# ignore line has to be exactly above line causing issues


RUN yum update -y
RUN yum install -y wget

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

# install in batch (silent) mode, does not edit PATH or .bashrc or .bash_profile
# -p path
# -f force
RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH=/root/miniconda3/bin:${PATH}

#RUN source /root/.bashrc
#RUN source /root/.bash_profile

# cleanup
RUN rm Miniconda3-latest-Linux-x86_64.sh


##############################################################
RUN conda create -n trading_env python=3.6 pip
SHELL ["conda", "run", "-n", "trading_env", "/bin/bash", "-c"]

# should work as well
#  conda init bash
#  source ~/.bashrc
#  conda activate trading_env


RUN python --version

# hadolint ignore=DL3013
RUN pip install --upgrade pip &&\
    pip install -r requirements.txt

RUN pip install cython
RUN pip install yfinance
RUN pip3 install yfinance
RUN pip install joblib
#RUN pip install pandas_datareader


RUN conda update -y --all

#RUN conda list
#RUN conda install -c anaconda jupyterlab
#RUN conda install -c anaconda scipy
#RUN conda install -c conda-forge fbprophet
#RUN conda install keras

RUN conda install -c quantopian ta-lib
RUN conda install -c anaconda pandas-datareader
RUN conda install -c conda-forge streamlit
RUN conda install -c conda-forge scikit-learn
RUN conda install -c conda-forge keras
RUN conda install -c conda-forge matplotlib
RUN conda install pytorch torchvision cpuonly -c pytorch
#RUN conda install keras
RUN conda install -c anaconda pandas
RUN conda install -c anaconda numpy


# have everything updated
#RUN conda update -y --all


## Step 4:
# Expose port 80, 8501
EXPOSE 80
EXPOSE 8501


## Step 5:
# Run app.py at container launch
#CMD ["streamlit" ,"run", "app.py"]
CMD ["conda", "run", "-n", "trading_env", "streamlit", "run", "app.py", "--server.port=80"]
