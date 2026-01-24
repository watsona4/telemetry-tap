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
    docker-cli \
    git

# Install telemetry-tap from GitHub (always pull latest from main)
ARG CACHE_BUST=2
RUN pip3 install --no-cache-dir --break-system-packages --upgrade \
    git+https://github.com/watsona4/telemetry-tap.git@main \
    && apk del git

# Copy run script
COPY run.sh /
RUN chmod a+x /run.sh

CMD [ "/run.sh" ]
