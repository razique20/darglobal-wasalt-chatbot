# Re-scrape the data: docker build -f scraper/Dockerfile -t prop-scraper . && docker run --rm -v "$PWD/data:/work/data" prop-scraper
FROM mcr.microsoft.com/playwright/python:v1.49.1-noble
WORKDIR /work
COPY scraper ./scraper
RUN pip install --no-cache-dir -r scraper/requirements.txt
CMD ["sh", "-c", "python scraper/wasalt.py && python scraper/darglobal.py"]
