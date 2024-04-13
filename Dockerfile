FROM ubuntu:23.10
MAINTAINER Nicolas Castel <nic.castel@gmail.com>

WORKDIR /app
RUN \
    apt-get update && \
    apt-get -y install \
    flatpak astap-cli darktable python3-pip python3-venv
RUN flatpak remote-add flathub https://dl.flathub.org/repo/flathub.flatpakrepo
RUN flatpak install -y flathub org.free_astro.siril

# Install python dependencies
RUN python3 -m venv /opt/venv
COPY requirements.txt .
RUN /opt/venv/bin/pip install -r requirements.txt

COPY . /app
EXPOSE 7860
CMD ["/opt/venv/bin/streamlit", "run", "app.py", "--server.port=7860", "--browser.gatherUsageStats", "false"]
