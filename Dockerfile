# Use an official Python image
FROM python:3.10.12

# Set the working directory
WORKDIR /app

# Copy the entire project, including the ml_model submodule
COPY . .

# Ensure submodules are initialized (if using Git submodules)
RUN git submodule update --init --recursive || true

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Start both FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn app.backend.app:app --host 0.0.0.0 --port 8000 & streamlit run app/frontend/app.py --server.port=8501 --server.address=0.0.0.0"]
