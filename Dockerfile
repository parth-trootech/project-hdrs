# Use an official Python runtime as a parent image
FROM python:3.10.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory (backend and frontend)
COPY app /app

# Expose the ports for both FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Command to run both FastAPI and Streamlit concurrently
CMD ["bash", "-c", "uvicorn app/backend/app.py --host 0.0.0.0 --port 8000 & streamlit run app/frontend/app.py"]
