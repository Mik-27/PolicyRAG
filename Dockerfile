# Use Python 3.10 as base image
FROM python:3.12.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Chrome
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg \
    unzip \
    git \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create directory structure
RUN mkdir -p /app/documents /app/chromedriver /app/utils /app/templates /app/static

# Install ChromeDriver that matches the Chrome version
RUN CHROME_VERSION=$(google-chrome --version | awk '{ print $3 }' | cut -d. -f1) \
    && CHROMEDRIVER_VERSION=$(curl -s "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_$CHROME_VERSION") \
    && echo "Chrome version: $CHROME_VERSION, ChromeDriver version: $CHROMEDRIVER_VERSION" \
    && wget -q --no-verbose -O /tmp/chromedriver.zip "https://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip" \
    && unzip /tmp/chromedriver.zip -d /app/chromedriver \
    && chmod +x /app/chromedriver/chromedriver \
    && rm /tmp/chromedriver.zip \
    && echo "ChromeDriver installed to /app/chromedriver/chromedriver"

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py ./
COPY templates/ ./templates/
COPY static/ ./static/
COPY utils/ ./utils/
# COPY .env .

# Install NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Create a script to start the application
RUN echo '#!/bin/bash\n\
    echo "Starting PolicyRAG application..."\n\
    python app.py\n' > /app/start.sh \
    && chmod +x /app/start.sh

# Expose port for Flask
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CHROMEDRIVER_PATH=/app/chromedriver/chromedriver

# Command to run the application
CMD ["/app/start.sh"]