FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pillow==10.0.0 \
    numpy==1.24.3 \
    torchvision==0.15.2

RUN mkdir -p /app

COPY ./model.pth /app
COPY ./reference1.png /app
COPY ./reference2.png /app
COPY ./main.py /app

ENV FLAG=myctf{town_in_go}

WORKDIR /app

EXPOSE 9999

CMD ["python3", "./main.py"]