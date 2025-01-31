# Docker

**Docker** is a containerization platform that enables the **packaging**, **distribution**, and **execution** of applications within **lightweight**, **isolated environments** called **containers**.

This tool is very useful because it ensures **consistency** across **different environments**.

## How we created a Docker Container

In the [Dockerfile](Dockerfile), we defined the container's behavior as follows:

```bash
FROM python:3.10
COPY requirements.txt requirements.txt 
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
WORKDIR /art2cook
COPY . .
RUN pip install .
EXPOSE 8080
CMD ["fastapi", "run", "src/api/art2mus_api.py", "--port", "8080"]
```

Following a concise explanation of the content of the Dockerfile:

We specify the **base image** as Python 3.10 available on [Python's Docker Hub page](https://hub.docker.com/_/python). Following that, we **upgrade pip**, **copy the entire current directory** into the container, and **install project dependencies** (requirements.txt). The working directory is switched to "/src" and, to enable external connections, we **expose port 8080** within the container. Finally, we state the **command** that runs the FastAPI server to expose Art2Cook's endpoints.

With the following command, we create **a Docker image** named "**art2cook**" with version "**1.0**," utilizing the current directory as the build context.

```bash
docker build -t art2cook:1.0 .
```

The resulting image is visible in [Docker Desktop](https://www.docker.com/products/docker-desktop/), as illustrated below:


Subsequently, we **launch the built image**, establishing a **mapping** between **host port 8080** and **container port 8080** for external application access.

```bash
docker run -p 8080:8080 art2cook:1.0
```

This procedure grants access to the Swagger UI, providing a user-friendly interface for interacting with our API. In our specific case, you can access the Swagger UI at:

```bash
http://localhost:8000/docs
```

More information about this tool can be found [here](https://docs.docker.com/). Other images can be found on [Docker Hub](https://hub.docker.com/).
