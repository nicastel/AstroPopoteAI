FROM ubuntu:23.10
MAINTAINER Nicolas Castel <nic.castel@gmail.com>

WORKDIR /app
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:lock042/siril
RUN \
    apt-get update && \
    apt-get -y install \
    darktable python3-pip python3-venv wget siril unzip

# Install python dependencies
RUN python3 -m venv /opt/venv
COPY requirements.txt .
RUN /opt/venv/bin/pip install -r requirements.txt
ADD https://gitlab.com/free-astro/pysiril/uploads/8224707c29669f255ad43da3b93bc5ec/pysiril-0.0.15-py3-none-any.whl pysiril-0.0.15-py3-none-any.whl
RUN /opt/venv/bin/pip install pysiril-0.0.15-py3-none-any.whl

# Astap cli and star dabase for plate solving
ADD https://github.com/nicastel/AstroPopoteAI/releases/download/astap/astap_command-line_version_Linux_aarch64.zip astap_command-line_version_Linux_aarch64.zip
RUN unzip astap_command-line_version_Linux_aarch64.zip
RUN wget -O d20_star_database.zip "https://drive.usercontent.google.com/download?id=1aCAKK0tB6eCNrzqPvwfCq-vg70ik0Ug4&export=download&authuser=0&confirm=t&uuid=a4415b42-70d3-4bd4-8025-7889ce24e518&at=APZUnTWIGJsa5pA0mQlPiX83UAgi%3A1713168442602"
RUN unzip d20_star_database.zip

COPY . /app
EXPOSE 7860
CMD ["/opt/venv/bin/streamlit", "run", "app.py", "--server.port=7860", "--browser.gatherUsageStats", "false"]
