# use slim image for smaller size
FROM python:3.11-slim

# set workdir
WORKDIR /app

# install system dependencies (sentence-transformers needs them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# copy requirements.txt and use docker cache for faster builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pre-download the sentence-transformers model to speed up the first run
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# copy project code
COPY . .

# streamlit default port is 8501
EXPOSE 8501

# health check (needed for Fargate)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# start Streamlit
# --server.address=0.0.0.0 makes it accessible from outside the container
# --server.headless=true prevents opening a browser
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]