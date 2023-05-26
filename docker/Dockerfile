FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04

RUN apt update && apt install python3 python3-pip -y

COPY requirements.txt /

RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN pip install -r /requirements.txt

RUN mkdir /.cache && chmod 777 /.cache 
