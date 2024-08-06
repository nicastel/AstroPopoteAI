FROM ubuntu:23.10
MAINTAINER Nicolas Castel <nic.castel@gmail.com>

WORKDIR /content/AstroPopoteAI
RUN apt-get update && apt-get install -y software-properties-common
# Add PPA for latest siril release
RUN add-apt-repository ppa:lock042/siril
# Add PPA for latest darktable release
RUN add-apt-repository ppa:ubuntuhandbook1/darktable
RUN \
    apt-get update && \
    apt-get -y install \
    python3-pip wget siril unzip xz-utils pkg-config libhdf5-dev python3-tk darktable

# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
ADD https://gitlab.com/free-astro/pysiril/uploads/8224707c29669f255ad43da3b93bc5ec/pysiril-0.0.15-py3-none-any.whl pysiril-0.0.15-py3-none-any.whl
RUN pip install pysiril-0.0.15-py3-none-any.whl

# Astap cli and star database for plate solving
ADD https://github.com/nicastel/AstroPopoteAI/releases/download/astap/astap_command-line_version_Linux_aarch64.zip astap_command-line_version_Linux_aarch64.zip
ADD https://github.com/nicastel/AstroPopoteAI/releases/download/astap/astap_command-line_version_Linux_amd64.zip astap_command-line_version_Linux_amd64.zip
# unzip the proper file for either aarch64 or amd64 architecture
#RUN /bin/bash -c 'set -ex && \
#    ARCH=`uname -m` && \
#    if [ "$ARCH" == "arm64" ]; then \
#    echo "aarch64" && \
#    unzip astap_command-line_version_Linux_aarch64.zip; \
#    else \
#    echo "unknown arch assuming amd64" && \
#    unzip astap_command-line_version_Linux_amd64.zip; \
#    fi'
RUN unzip astap_command-line_version_Linux_aarch64.zip
RUN wget -O d20_star_database.zip "https://drive.usercontent.google.com/download?id=1aCAKK0tB6eCNrzqPvwfCq-vg70ik0Ug4&export=download&authuser=0&confirm=t&uuid=a4415b42-70d3-4bd4-8025-7889ce24e518&at=APZUnTWIGJsa5pA0mQlPiX83UAgi%3A1713168442602"
RUN unzip d20_star_database.zip

# Stars catalog for siril
ADD https://free-astro.org/download/kstars-siril-catalogues/namedstars.dat.xz namedstars.dat.xz
RUN unxz namedstars.dat.xz
ADD https://free-astro.org/download/kstars-siril-catalogues/unnamedstars.dat.xz unnamedstars.dat.xz
RUN unxz unnamedstars.dat.xz
ADD https://free-astro.org/download/kstars-siril-catalogues/deepstars.dat.xz deepstars.dat.xz
RUN unxz deepstars.dat.xz
ADD https://free-astro.org/download/kstars-siril-catalogues/USNO-NOMAD-1e8.dat.xz USNO-NOMAD-1e8.dat.xz
RUN unxz USNO-NOMAD-1e8.dat.xz

# Graxpert
ADD https://github.com/Steffenhir/GraXpert/archive/refs/tags/3.0.2.zip GraXpert-3.0.2.zip
RUN unzip GraXpert-3.0.2.zip

# Starnet
ADD https://github.com/nicastel/starnet/releases/download/starnetv1/starnet_weights2.zip starnet_weights2.zip
RUN unzip starnet_weights2.zip

# Darktable style folder creation
RUN mkdir -p /root/.config/darktable/styles

COPY . /content/AstroPopoteAI
RUN chmod +x /content/AstroPopoteAI/run_starnet.sh
COPY s3_secrets.py /content/AstroPopoteAI/GraXpert-3.0.2/graxpert/
COPY astro.dtstyle /root/.config/darktable/styles

EXPOSE 7860
CMD ["/opt/venv/bin/streamlit", "run", "app.py", "--server.port=7860", "--browser.gatherUsageStats", "false"]
