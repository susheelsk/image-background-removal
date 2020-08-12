FROM python:3.7.8
# Setup workdir
WORKDIR /usr/src/app
# Copy files
COPY requirements_http.txt ./
COPY . .
# Dependency Installation
RUN pip3 install --no-cache-dir -r requirements_http.txt
# Download Models
RUN (python3 setup.py --model all)
# Setting environment variables
ENV HOST='0.0.0.0'
ENV PORT='5000'
ENV AUTH='false'
ENV MODEL='u2net'
ENV PREPROCESSING='None'
ENV POSTPROCESSING='fba'
ENV ADMIN_TOKEN='admin'
ENV ALLOWED_TOKENS_PYTHON_ARR="['test']"
ENV IS_DOCKER_CONTAINER='true'

EXPOSE 5000

ENTRYPOINT ["python3", "./http_api.py"]
