import os
import sys
import torch
from http import HTTPStatus 
from functools import wraps
from api_schema import DataPayload
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

CURR_FILE_PATH = os.path.abspath(__file__)
PROJ_DIR = os.path.join(CURR_FILE_PATH.split('art2cook')[0], 'art2cook')
sys.path.append(PROJ_DIR)

from test_art2mus import generate_music


# Additional details for each endpoint
tags_metadata = [
    {
        "name": "Root",
        "description": "Explore the root endpoint for essential details, including version, authors, and external links.",
    },
    {
        "name": "Generate_Music",
        "description": "Generate new music based on a digitized artwork!",
    }

]

APP_DESCRIPTION_MESSAGE = (
    "Welcome! Explore **Art2Mus**, an **Artwork-Based Music Generation System**! "
    "Feel free to test the API's endpoints with the default artworks or your own!"
    )


app = FastAPI(
    title="Art2Mus",
    description=APP_DESCRIPTION_MESSAGE,
    version="v01",
    openapi_tags=tags_metadata
)

# Need this to let the two localhost servers to interact with each other (API and frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
        }

        # Additional data in the response
        if "data" in results:
            response["data"] = results["data"]

        if "version" in results:
            response["version"] = results["version"]

        if "authors" in results:
            response["authors"] = results["authors"]

        if "github" in results:
            response["github"] = results["github"]
            
        if "path_to_music" in results:
            response["path_to_music"] = results["path_to_music"]

        return response

    return wrap


@app.get("/", tags=["Root"])
@construct_response
def _index(request: Request):
    """Root endpoint.

    **Parameters**
    - No parameters needed

    **Output**
    - A **JSON object** containing the:
        - **HTTP message**
        - the **HTTP status code**
        - a **welcome message*
        - the **names of the system's authors**
    """

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to Art2Mus! Please, read the `/docs` if you want to use our system!"},
        "version": "Current: 1.1",
        "authors": ['Ivan Rinaldi', 'Nicola Fanelli', 'Giovanna Castellano', 'Gennaro Vessio'],
        'github': 'https://github.com/colonnaemanuele/art2cook',
    }

    return response


@app.post("/music-generation/", tags=["Generate_Music"])
@construct_response
def _generate_music(request: Request, data_payload: DataPayload):
    """
    Endpoint to **generate music** based on the current artwork.

    **Parameters**
    - **Path** to the digitized artwork to use to generate music

    **Output**
    - If everything works out, a **JSON object** containing the path to the generated music
    - Otherwise, an **exception** will be raised
    """

    print(data_payload)

    if not data_payload.digitized_artwork_path:
        # If no path is found, raise an exception
        raise HTTPException(status_code=404, detail='We are sorry, but the path to the digitized artwork is missing.')
    else:
        
        # Otherwise, generate music based on the provided artwork
        path_to_music, is_music_stored = generate_music(data_payload.digitized_artwork_path, music_file_name="art2mus_example_api.wav")
        torch.cuda.empty_cache()

        if is_music_stored:
            response = {
                "message": HTTPStatus.OK.phrase,
                "status-code": HTTPStatus.OK,
                "path_to_music": path_to_music,
                "data": {"message": "Music generated successfully!"}
            }
            return response
        
        else:
            raise HTTPException(status_code=404, detail='We are sorry, our system was not able to generate music.')