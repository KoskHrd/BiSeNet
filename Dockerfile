FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN conda install -y tabulate
RUN pip install opencv-python
RUN pip install -U albumentations
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-dev

