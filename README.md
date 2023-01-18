![Inspiring Image](hhttps://repository-images.githubusercontent.com/555775869/e680cc8c-c58b-4c76-8ce6-08dd07c2a4d5)

# Caelvid
Caelvid comes from the latin "Caelum Videre", which means "Look at the Sky". It is a library including several computer vision models

# Object Detection

## Yolo

### Installation

##### Set up Environment Variables
Set up an environment variables called `YOLO_OBJECT_DETECTION_PATH` that hosts the path to the `./object_detection/yolo`
folder.

### Running API Service
Change directory into `object_detection/yolo` and run the following command:
``` bash
uvicorn packages.rest_api.rest_api:app --reload
```

### Docker

#### Build
The Dockerfile is located in the `docker/detect_single_object.Dockerfile` and it is possible to build it through the following command:

``` bash
docker image build -t <repository>/<image_name>:<tag> -f detect_single_object.Dockerfile .
```