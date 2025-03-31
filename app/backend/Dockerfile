# Use an official Python image as the base
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the application files
COPY ./app /app

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
