FROM transformers:ds
LABEL maintainer="BELLE"
WORKDIR /workspace

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt update
RUN apt install -y git-lfs
RUN apt install -y htop
RUN apt install -y screen
RUN apt install -y tmux
RUN apt install -y locales
RUN locale-gen en_US.UTF-8
RUN locale-gen zh_CN.UTF-8
RUN apt install -y net-tools
RUN apt install -y openssh-server \
    && sed -i "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/" /etc/ssh/sshd_config \
    && sed -i "s/#PubkeyAuthentication yes/PubkeyAuthentication yes/" /etc/ssh/sshd_config \
    && sed -i "s/#PasswordAuthentication yes/PasswordAuthentication no/" /etc/ssh/sshd_config \
    && echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config \
    && mkdir -p /run/sshd

RUN python3 -m pip install -U --no-cache-dir pip
RUN python3 -m pip install -U --no-cache-dir peft
RUN python3 -m pip install -U --no-cache-dir gradio
RUN python3 -m pip install -U --no-cache-dir pudb
RUN python3 -m pip install -U --no-cache-dir xformers
RUN python3 -m pip install -U --no-cache-dir bitsandbytes
RUN python3 -m pip install -U --no-build-isolation --no-cache-dir flash-attn
RUN python3 -m pip install -U --no-cache-dir install git+https://github.com/wookayin/gpustat.git@master
RUN python3 -m pip install -U --no-cache-dir ipykernel
RUN python3 -m pip install -U --no-cache-dir ipywidgets
RUN python3 -m pip install -U --no-cache-dir httpx[socks]
RUN python3 -m pip install -U --no-cache-dir wandb
RUN echo -e 'export LANG=zh_CN.UTF-8' >> /root/.bashrc

RUN mkdir -p /scripts && echo -e '#!/bin/bash\n\
SSHD_PORT=22001\n\
CMD_TO_RUN=""\n\
while (( "$#" )); do\n\
  case "$1" in\n\
    --sshd_port)\n\
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then\n\
        SSHD_PORT=$2\n\
        shift 2\n\
      else\n\
        echo "Error: Argument for $1 is missing" >&2\n\
        exit 1\n\
      fi\n\
      ;;\n\
    --cmd)\n\
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then\n\
        CMD_TO_RUN=$2\n\
        shift 2\n\
      else\n\
        echo "Error: Argument for $1 is missing" >&2\n\
        exit 1\n\
      fi\n\
      ;;\n\
    -*|--*=) \n\
      echo "Error: Unsupported flag $1" >&2\n\
      exit 1\n\
      ;;\n\
    *) \n\
      shift\n\
      ;;\n\
  esac\n\
done\n\
sed -i "s/#Port 22/Port $SSHD_PORT/" /etc/ssh/sshd_config\n\
/usr/sbin/sshd\n\
if [ -n "$CMD_TO_RUN" ]; then\n\
  bash -c "$CMD_TO_RUN"\n\
else\n\
  /bin/bash\n\
fi' > /scripts/startup.sh && chmod +x /scripts/startup.sh

ENTRYPOINT ["/bin/bash", "/scripts/startup.sh"]
