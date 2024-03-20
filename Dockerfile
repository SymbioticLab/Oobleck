FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace
ENV PYTHONPATH=/workspace

RUN apt update -y && apt install wget git -y && apt clean

RUN conda update -n base --override-channels -c defaults conda
RUN conda install -y \
    numpy \ 
    scikit-learn \
    pytest \
    pytest-mock \
    pytest-asyncio \
    tensorboard \
    glpk \
    setuptools \
    pybind11 \
    ninja \
    cmake \
    tbb-devel \
    conda-forge::cyipopt \
    conda-forge::python-devtools \
    conda-forge::pyomo \
    conda-forge::cxx-compiler \
    "conda-forge::transformers>=4.29.0" \ 
    "conda-forge::deepspeed>=0.8.1" \
    conda-forge::accelerate \
    conda-forge::datasets \
    conda-forge::evaluate
  
RUN pip install psutil simple-parsing asyncssh aiofiles

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t https://github.com/denysdovhan/spaceship-prompt \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions
RUN conda init zsh
RUN chsh -s /bin/zsh root
CMD [ "/bin/zsh" ]