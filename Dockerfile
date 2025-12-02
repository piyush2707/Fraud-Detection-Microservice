# Use a lightweight official Python image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the saved model and scaler assets
COPY model/ ./model/

# Copy the API code
COPY api.py .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the Uvicorn server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

