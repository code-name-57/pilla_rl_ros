# Use the official ROS 2 Humble base image
FROM ros:humble AS main

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
    ros-humble-joy \
    && rm -rf /var/lib/apt/lists/*

ARG GPU_BUILD=true

# Set the working directory inside the container
WORKDIR /ros2_ws/src

COPY requirements* .

RUN uv venv --python 3.10 && \
    . .venv/bin/activate && \
    if [ "$GPU_BUILD" = "true" ]; then \
        uv pip install -r requirements.gpu.txt --extra-index-url https://download.pytorch.org/whl/cu126; \
    else \
        uv pip install -r requirements.cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY launch/ launch
COPY resource/ resource/
COPY package* setup* .
COPY pilla_rl_ros/ pilla_rl_ros/

RUN . /opt/ros/humble/setup.sh && \
    colcon build

SHELL ["/bin/bash", "-c"]
ENTRYPOINT . /opt/ros/humble/setup.bash && \
           . install/setup.bash && \
           . .venv/bin/activate && \
           ros2 launch pilla_rl_ros pilla_rl.launch.py