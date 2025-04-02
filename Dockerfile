# Use an official Python image
FROM python:3.10.12

# Set the working directory
WORKDIR /app

# Install system dependencies required by OpenCV and PostgreSQL
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the wheelhouse directory and requirements.txt first
COPY wheelhouse /wheelhouse
COPY requirements.txt .

# Install Python dependencies from the wheelhouse
RUN pip install --no-index --find-links=/wheelhouse -r requirements.txt

# Copy the rest of the project
COPY . .

# Ensure Git submodules are initialized (if using Git submodules)
RUN git submodule update --init --recursive || true

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Start both FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn app.backend.app:app --host 0.0.0.0 --port 8000 & streamlit run app/frontend/app.py --server.port=8501 --server.address=0.0.0.0"]
