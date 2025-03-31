# Use an official Python image as the base
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the application files
COPY ./app /app

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app/frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
