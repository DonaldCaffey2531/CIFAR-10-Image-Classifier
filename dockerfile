# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy code and model
COPY app.py ./app.py
COPY saved_model ./saved_model

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn tensorflow numpy pydantic

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
