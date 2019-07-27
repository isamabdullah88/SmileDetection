# Simile Detection

This repo contains a simple implementation of smile detection algorithm based on dlib facial points captured.

## Getting Started
This repository uses opencv and dlib libraries in python.

### Prerequisites
Following are the required libraries for repo.
```bash
opencv
numpy-1.16
cmake (python)
dlib
```
### Installing
A virtual environment is highly recommended to install these packages. That will make the installation process clean and isolated. 
```bash
pip install opencv-python
pip install numpy
pip install cmake
pip install dlib
```

## Usage
- Please note the this repo is only tested on Windows 10.
- To use first clone the repository.
- Please run the file **smileDetector.py**.

### Running
- Download the trained model from this [link](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). Place the downloaded file in **Models/** folder in the project.
- There is a commandline arguemnt for the index of webcam. Normally it is 0 (which is the default). If it does not work for you, please change it using a different number (say 1):
```
python smileDetector.py --webcam_idx 1
```


## Experimentation
### Details
- Get face rectangle using dlib model.
- Get the 68 points describing the detailed shape of face using the dlib shape predictor.
- Compute if the face is smiling or not using points around mouth.
- Post-process the per frame results by applying a windowing filter to make the results more consistent.
### Results
- Gives a reasonable accuracy for smile detection.

### Scaling
-  For scaling it to multiple cameras, the process has to be parallelized.
- The computational resources have to be carefully managed and checked if these are suitable for large scale.

## Pull Requests
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[GNU](https://github.com/isamabdullah88/VehicleClassification/blob/master/LICENSE)


