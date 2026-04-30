# 用 slim 镜像, 体积小
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 装系统依赖 (sentence-transformers 需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 先复制 requirements 利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预下载 sentence-transformer 模型到镜像 (避免容器启动时下载)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 复制项目代码
COPY . .

# Streamlit 默认端口
EXPOSE 8501

# 健康检查 (Fargate 用得上)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 启动 Streamlit
# --server.address=0.0.0.0 让容器外能访问
# --server.headless=true 不弹浏览器
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]