FROM python:3.7.8

WORKDIR /usr/src/app

COPY requirements.txt ./
COPY requirements_http.txt ./

RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -r requirements_http.txt


COPY . .

RUN (cd tools && python3 setup.py all)



ENTRYPOINT ["python3", "./http_api.py", "-m" ,"u2net", "-postp", "rtb-bnb", "-prep", "bbd-fastrcnn", "-auth", "false", "-port", "5000", "-host", "0.0.0.0"]
