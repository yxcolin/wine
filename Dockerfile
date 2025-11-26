FROM python:3.9-slim

# 设置环境变量
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖（使用国内镜像加速）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 复制所有项目文件
COPY . .

# 设置启动命令
CMD ["python", "train.py"]
