# License Plate detection and extraction with YOLO and PaddleOCR

This project implements a YOLOv11 trained model to detect vehicle license plates. Then, with PaddleOCR and a specific preprocessing programmed the vehicle registration is extracted from the license plate. The project also provides the option
to make de detection and extraction from an image or in real time with a functional camera.

![](/assets/ext.png)

## Features
- **Detection:** the provided YOLO trained model detects license plates in vehicles. First recognizing the frontal car structure and then detecting the license plate from it.
- **Extraction:** the project contains an OCR that extract the vehicle registration by applying a programmed filter.
- **Modes:** the project contains 2 different types of detection and extraction: in static images or real time.

## Table of contents 
- [Requirements](#Requirements)
- [Installation](#Installation)
- [Operation](#Operation)
- [Specs](#Specs)
- [Project Structure](#Project-Structure)
- [Troubleshooting](#Troubleshooting)
- [Contributing](#Contributing)

## Requirements
**Hardware**
- Webcam

**Software**
- Ubuntu **22.04** 
- Python 3.10 / Window 10 >

## Installation

The project contains two scripts. So, **images_ext.py** analyzes images defined in a specific path. Then, imports de YOLO trained model to detect the license plates inside the images. After that, using OpenCV the license plate is cropped and processed, to prepare the cropped image to be analyzed by the OCR. The preprocessing consists in crop the specific box that contains the license plate, and then change it to a gray scale. It also apply morphological operations to clean the noise from the extraction operation. Finally, the OCR by PaddleOCR is applyed to extract the desired text, looking for one of the defined patterns in the code.

The repository also contains **realtime_ext.py**, this script has the same function as the previous one. But it also make detections and extractions in real time, so other configurations and parameters were neccesary to apply, such as clenup image at the end of the process, detections by frame configuration and an applyed masking to despise noise and unnecessary words in the extractions.

Finally it contains **best.pt**, that consists in a YOLO model **version 11**. Due to the model was previously trained, it can be used without training it again. If you are interested in training your own model, here is the link to Roboflow project that contains a robust dataset for your own model training.

```bash
https://universe.roboflow.com/licenseplate-s6fjf/license-plate-xmnzu
```
So first, let's clon this repository by opening the terminal and writing:

```bash
git clone https://github.com/aTrujillo04/License-plate-detection-and-extraction.git
```

Then, let's download the necessary tools to run and test the project:

```bash
cd /route/to/this/repository
python3 -m venv name
source name/bin/activate
```
The previous lines place you in the repository location inside your computer, create your virtual enviroment and activate it. When the virtual enviroment is activated you should see something like this:

```bash
(name) user@computer:~/route/to/repository$
```

Now let's download the requirements **inside the new virtual enviroment** and verify the installation by seeing the downloaded version:

```bash
pip install -r requirements.txt
pip freeze
```
By the last command you should be able to see a list of necessary tool to test the project, such as this:

```bash
ackermann-msgs==2.0.2
action-msgs==1.2.2
action-tutorials-interfaces==0.20.5
...
```

## Operation

Now, let's prepare the necessary resources and configurations to get a correct script launch.

So, let's start the **realtime_ext.py**. To run this script a folder needs to be created inside the project folder. This new folder will contain **the license plates images** you want to detect and 
extract. It is suggested to take the photos in a frontal plane or not too inclined plane, the OCR won't work with too inclined planes. find here some examples:

IMAGENES EDITADASSSSSSSSSS

LINEASSSS A CAMBIAR PARA CORRER

Then, you wil be able to **launch the scripts** by entering the following commands in the terminal:

```bash
python3 realtime_ext.py
```

```bash
python3 images_ext.py
```

If the both scripts run optimally you should be able to see something like this:

```bash
 @app.on_event("startup")
INFO:     Started server process [2242]
INFO:     Waiting for application startup.
Hardware initialized
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
```
