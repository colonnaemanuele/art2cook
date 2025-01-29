## Run Art2Mus on your local machine 🖼️🎵

> [!IMPORTANT]
> The code should work regardless of whether **CUDA** is installed on your machine. However, please note that inference will take longer on a CPU compared to a GPU.
> of image transformations on the final generated music.

To run the local server using FastAPI, make sure you're in the project's directory. If you're not, navigate to it with the following command:

```bash
cd art2cook/src/api
```

Once you're in the correct directory, start FastAPI's local server with this command:

```bash
uvicorn art2mus_api:app --reload --port 8001
```

> [!NOTE]
> The --port is included because uvicorn's default port (8000) is already in use in our setup. If this is not the case for your environment, you can omit it.
