---
title: Docker 操作
date: 2025-01-20 19:00:00
tags:
- docker 
- 开发
categories:
- 开发笔记
alias:
- developnotes/docker/
---

### 创建镜像
通常，创建 Docker 镜像的方法是通过 `Dockerfile` 文件。`Dockerfile` 是一个文本文件，包含了构建镜像所需的所有指令。

##### 基于 Ubuntu 创建 Python 环境
```python
# 使用官方 Ubuntu 镜像作为基础镜像
FROM ubuntu:20.04

# 设置环境变量，防止在安装过程中出现交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 更新镜像并安装 Python 和 pip
RUN apt-get update && apt-get install -y python3 python3-pip

# 设置工作目录
WORKDIR /app

# 将本地目录的代码复制到容器内
COPY . /app

# 安装 Python 依赖（假设有 requirements.txt 文件）
RUN pip3 install -r requirements.txt

# 默认运行命令
CMD ["python3", "app.py"]
```

##### 基于 Python 官方镜像
```python
# 使用官方 Python 3.9 镜像作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录内容到容器中的工作目录
COPY . /app

# 安装 Python 依赖
RUN pip install -r requirements.txt

# 运行 Python 应用
CMD ["python", "app.py"]
```

在包含 `Dockerfile` 文件的目录中运行以下命令来构建镜像：

```shell
docker build -t mylocalenv:latest .
```
- `-t mylocalenv:latest` 用于为镜像指定一个标签（这里是 `mylocalenv`）。
- `.` 表示当前目录是构建上下文。


### 拉取现有镜像
`docker pull [OPTIONS] NAME[:TAG|@DIGEST]`  // **拉取镜像(下载)**

- NAME: 镜像名称，通常包含注册表地址（如 [docker.io/library/ubuntu）](http://docker.io/library/ubuntu%EF%BC%89)
- TAG(可选)：镜像标签
- DIGEST(可选)：镜像SHA256摘要

例如：`docker pull continuumio/anaconda3` 从网上拉最常用的 Anaconda3 镜像


### 启动容器并进入环境
常用指令：`docker run -it -v /mnt:/workspace/data my_env bash` 
- `-v` 代表挂在本地盘到容器中，让容器能访问本地磁盘
- `/mnt` 是本地磁盘路径，`/workspace/data` 是容器内部路径
- `bash` 代表我们希望用bash，看情况而加，若报错则不加
- `my_env` 是镜像名字

### 进入容器

`docker exec -it mycontainer bash`
`docker images` 列出镜像列表
`docker ps -a` 列出所有容器，`-a` 代表所有
`docker start mycontainer` 启动容器
`docker stop mycontainer` 停止容器
`docker kil mycontainer` 杀死容器
`docker rm mycontainer` 删除容器
`docker rmi myimage` 删除镜像（需删除关联容器）
`Ctrl + D` 退出容器

### 保存容器为新镜像

`docker commit -a 'author' -m 'instruction' mycontainer new_image` 保存为新镜像
`docker save -o tar_name.tar image_name` 将镜像保存为压缩包
`docker load -i tar_name.tar` 读取压缩包镜像，然后用 `docker images` 就能看到一个新镜像

### 本地 文件/环境 传入容器
`docker cp /home/b/miniconda3/envs/py39 mycontainer:/opt/conda/envs`
将本地的`py39`环境复制到容器`mycontainer`中