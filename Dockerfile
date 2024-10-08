FROM python:3.9-slim
WORKDIR /workspace
COPY . .
RUN apt-get update && apt-get install -y \
build-essential \
libssl-dev \
libffi-dev \
python3-dev
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 80
ENV NAME MLOpsLab
CMD ["python", "Train.py"]
