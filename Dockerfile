# Use official Python image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Install sudo and required system dependencies
RUN apt-get update && apt-get install -y sudo

# Create a non-root user and add it to sudoers
RUN useradd -m -s /bin/bash appuser && echo "appuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Switch to non-root user
USER appuser

# Copy necessary files
COPY --chown=appuser:appuser requirements.txt ./
COPY --chown=appuser:appuser app.py htmlTemplates.py LICENSE README.md ./

# Install dependencies using sudo
RUN sudo pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
