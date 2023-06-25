FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

COPY ./environment.yml /

WORKDIR /

RUN apt update -y && apt install wget git -y && apt clean
RUN wget -q https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh -O ./anaconda.sh && \
    /bin/bash ./anaconda.sh -b -p /opt/conda && \
    rm ./anaconda.sh && \
    /opt/conda/bin/conda init bash && \
    /opt/conda/bin/conda env create -f environment.yml && \
    /opt/conda/bin/conda clean -afy && \
    echo "conda activate oobleck" >> ~/.bashrc && \
    rm ./environment.yml