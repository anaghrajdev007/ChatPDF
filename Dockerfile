
FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y sudo

RUN useradd -m -s /bin/bash appuser && echo "appuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER appuser

COPY --chown=appuser:appuser requirements.txt ./
COPY --chown=appuser:appuser app.py htmlTemplates.py LICENSE README.md pdf_ai.py ./
COPY ./static ./static

# disable no cache for testing purposes
RUN sudo pip install -r requirements.txt

EXPOSE 8501

CMD ["python", "app.py"]
