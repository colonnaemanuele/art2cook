FROM python:3.10

COPY requirements.txt requirements.txt 

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

WORKDIR /art2cook

COPY . .

RUN pip install .

# Expose port
EXPOSE 8080

# Run the FastAPI application
CMD ["fastapi", "run", "src/api/art2mus_api.py", "--port", "8080"]