![Inspiring Image](https://repository-images.githubusercontent.com/555775869/e680cc8c-c58b-4c76-8ce6-08dd07c2a4d5)

# Caelvid
Caelvid comes from the latin "Caelum Videre", which means "Look at the Sky". It is a library including several computer vision models

# Installation

## Update PYTHONPATH
Add the current directory to the `PYTHONPATH` environment variables.
``` bash
export PYTHONPATH="$PYTHONPATH:/<absolute_path>/caelvid"
```


## Running API Service
Change directory into `object_detection/yolo` and run the following command:
``` bash
uvicorn src.object_detection_yolov3.object_detection_rest_api:app --reload
```

## Docker

### Build
The Dockerfile is located in the `docker/detect_single_object.Dockerfile` and it is possible to build it through the following command:

``` bash
# Change directory to parent one
docker image build --progress=plain -f ./docker/detect_single_object.Dockerfile -t <repository>/<image_name>:<tag> . 
```