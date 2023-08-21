FROM belle

# https://docs.nvidia.com/networking/m/view-rendered-page.action?abstractPageId=15049785
ENV MOFED_VER=23.07-0.5.0.0
ENV OS_VER=ubuntu20.04
ENV PLATFORM=x86_64

RUN wget http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}/MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    tar -xvf MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}/mlnxofedinstall --user-space-only --without-fw-update -q