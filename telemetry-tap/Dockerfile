ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install Python and dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-psutil \
    smartmontools \
    lm-sensors \
    dmidecode \
    docker-cli

# Install telemetry-tap from PyPI or GitHub
RUN pip3 install --no-cache-dir --break-system-packages \
    git+https://github.com/watsona4/telemetry-tap.git

# Copy run script
COPY run.sh /
RUN chmod a+x /run.sh

CMD [ "/run.sh" ]
