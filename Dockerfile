FROM ubuntu:23.10
MAINTAINER Nicolas Castel <nic.castel@gmail.com>

WORKDIR /app
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:lock042/siril
RUN \
    apt-get update && \
    apt-get -y install \
    astap-cli darktable python3-pip python3-venv wget siril

# Install python dependencies
RUN python3 -m venv /opt/venv
COPY requirements.txt .
RUN /opt/venv/bin/pip install -r requirements.txt
ADD https://gitlab.com/free-astro/pysiril/uploads/8224707c29669f255ad43da3b93bc5ec/pysiril-0.0.15-py3-none-any.whl pysiril-0.0.15-py3-none-any.whl
RUN /opt/venv/bin/pip install pysiril-0.0.15-py3-none-any.whl

COPY . /app
EXPOSE 7860
CMD ["/opt/venv/bin/streamlit", "run", "app.py", "--server.port=7860", "--browser.gatherUsageStats", "false"]
