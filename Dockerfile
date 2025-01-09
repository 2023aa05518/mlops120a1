# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /src

# Copy files
COPY requirements.txt .
COPY src/app.py .
COPY best_model.pkl .

# Install dependencies
RUN pip install -r requirements.txt

# Expose Flask app port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]