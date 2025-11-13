# License Plate detection and extraction with YOLO and PaddleOCR

This project implements a YOLOv11 trained model to detect vehicle license plates. Then, with PaddleOCR and a specific preprocessing programmed the vehicle registration is extracted from the license plate. The project also provides the option
to make the detection and extraction from an image or in real time with a functional camera.

![](/assets/example2.jpeg)


## Features
- **Detection:** the provided YOLO trained model detects license plates in vehicles. First recognizing the frontal car structure and then detecting the license plate from it.
- **Extraction:** the project contains an OCR that extract the vehicle registration by applying a programmed filter.
- **Modes:** the project contains 2 different types of detection and extraction: in static images or real time.

## Table of contents 
- [Requirements](#Requirements)
- [Installation](#Installation)
- [Operation](#Operation)
- [Project Structure](#Project-Structure)
- [Troubleshooting](#Troubleshooting)
- [Contributing](#Contributing)

## Requirements
**Hardware**
- Functional PC/Laptop
- Webcam

**Software**
- Ubuntu **22.04** / Windows 10 >
- Python 3.10 

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

Now, let's prepare the necessary resources and configurations to get a correct **image_ext.py** script launch.

So, let's start the **image_ext.py**. To run this script a folder needs to be created inside the project folder. This new folder will contain **the license plates images** you want to detect and 
extract. It is suggested to take the photos in a frontal plane or not too inclined plane, the OCR won't work with too inclined planes. Find here some examples:

![](/assets/example.jpeg)

It is also neccesary to **change the image path** in **line 10** in **image_ext.py** script:

```bash
image = cv2.imread("./images_folder/image.jpeg")
```

Then, you wil be able to **launch the script** by entering the following commands in the terminal:

```bash
python3 images_ext.py
```

And also, in case of **images_ext.py** three window will deploy, one for the **image with detection and extraction applyed** and the other two images with the **cropped license plate and cropped license plate but in gray scale**. 

Then, for the real time detection and extraction, two important parameters have to be defined in **realtime_ext.py** script to run optimally:

First, in **line 26** you must define the listed device in which the video output is located on your local PC/Laptop:

```bash
video_source = X
```
To know in wich device the video output is lieted, do in Ubuntu terminal:

```bash
sudo apt update
sudo apt install v4l-utils
```

Once you downloaded the package do:

```bash
v4l2-ctl --list-devices
```

You may see something like this:

```bash
USB2.0 HD UVC WebCam: USB2.0 HD (usb-0000:36:00.3-4):
	/dev/video0
	/dev/video1
	/dev/media0

Depstech webcam: Depstech webca (usb-0000:36:00.4-1):
	/dev/video2
	/dev/video3
	/dev/media1
```

Analyzing the log, we see **Deepstech webcam** as the coneected webcam via USB, due to USB.0 HD is the **integrated webcam** in the Laptop we do not want to use. The video output is also listed as device 2: **/dev/video2**.

Now, it is important to search the camera resolution and specify the data correctly, in order to ensure a good operation and a resoruces saving. Also it will enhance the detection and extraction accuracy due to the frame processing.
So, after figuring out the exactly camera resolution, change it in **line 34 and 35**.

```bash
cap.set(cv2.CAP_PROP_FRAME_WIDTH, XXXX)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, XXX)}
```

Now, after setting this parameters. The script **realtime_ext.py** is ready to be launched by:

```bash
python3 realtime_ext.py
```

Finally, you will see a deployed window. Through this window you will be able to see the **real time camera record** and the **detection and extraction. 

If the both scripts run optimally you should be able to see something like this:

```bash
Creating model: ('PP-LCNet_x1_0_doc_ori', None)
Model files already exist. Using cached files. To redownload, please delete the directory manually: `/home/antoniotru/.paddlex/official_models/PP-LCNet_x1_0_doc_ori`.
Creating model: ('UVDoc', None)
Model files already exist. Using cached files. To redownload, please delete the directory manually: `/home/antoniotru/.paddlex/official_models/UVDoc`.
Creating model: ('PP-LCNet_x1_0_textline_ori', None)
Model files already exist. Using cached files. To redownload, please delete the directory manually: `/home/antoniotru/.paddlex/official_models/PP-LCNet_x1_0_textline_ori`.
Creating model: ('PP-OCRv5_server_det', None)
Model files already exist. Using cached files. To redownload, please delete the directory manually: `/home/antoniotru/.paddlex/official_models/PP-OCRv5_server_det`.
Creating model: ('en_PP-OCRv5_mobile_rec', None)
Model files already exist. Using cached files. To redownload, please delete the directory manually: `/home/antoniotru/.paddlex/official_models/en_PP-OCRv5_mobile_rec`.

0: 576x736 1 0, 1 2, 36.4ms
Speed: 4.1ms preprocess, 36.4ms inference, 1.3ms postprocess per image at shape (1, 3, 576, 736)
ultralytics.engine.results.Boxes object with attributes:
```

## Project Structure

```text
License-plate-detection-and-extraction/
├── assets/
│   ├── example.jpeg
│   ├── example1.jpeg
│   └── example2.jpeg
├── best.pt
├── gitignore
├── README.md
├── realtime_ext.py
├── requirements.txt
├── image_ext.py
```

## Troubleshooting

During the **image_ext.py** you may face two different problems:

 If you are trying to run the **image_ext.py** script and you receive the following output:

```bash
[ WARN:0@5.742] global loadsave.cpp:275 findDecoder imread_('./test_plates/test34.jpeg'): can't open/read file: check file path/integrity
Traceback (most recent call last):
  File "/home/antoniotru/veli_plate/images_ext.py", line 13, in <module>
    orig_h, orig_w, _ = image.shape
AttributeError: 'NoneType' object has no attribute 'shape'
```

It means that the image's path is not correctly defined, so you must copy and paste the exactly image path to solve the trouble. You also have to ensure that the image has a **jpeg, jpg or png** format.

Other common error output could be:

```bash
Traceback (most recent call last):
  File "/home/antoniotru/veli_plate/images_ext.py", line 16, in <module>
    model = YOLO("best") # Trained YOLO model for license plate detection
  File "/home/antoniotru/veli_plate/virenv/lib/python3.10/site-packages/ultralytics/models/yolo/model.py", line 81, in __init__
    super().__init__(model=model, task=task, verbose=verbose)
  File "/home/antoniotru/veli_plate/virenv/lib/python3.10/site-packages/ultralytics/engine/model.py", line 149, in __init__
    self._load(model, task=task)
  File "/home/antoniotru/veli_plate/virenv/lib/python3.10/site-packages/ultralytics/engine/model.py", line 293, in _load
    weights = checks.check_file(weights)  # runs in all cases, not redundant with above call
  File "/home/antoniotru/veli_plate/virenv/lib/python3.10/site-packages/ultralytics/utils/checks.py", line 577, in check_file
    raise FileNotFoundError(f"'{file}' does not exist")
FileNotFoundError: 'best' does not exist
```

This terminal output means that you are **not** loading the YOLO trained model **correctly**. So you must ensure to enter the corret model name including **.pt** extension. You also should verify that the YOLO model path is the correct one.

Now, for the **realtime_ext.py** script you may receive a **terminal error output**:

```bash
/home/antoniotru/veli_plate/virenv/lib/python3.10/site-packages/paddle/utils/cpp_extension/extension_utils.py:718: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
  warnings.warn(warning_message)
Creating model: ('PP-LCNet_x1_0_doc_ori', None)
Model files already exist. Using cached files. To redownload, please delete the directory manually: `/home/antoniotru/.paddlex/official_models/PP-LCNet_x1_0_doc_ori`.
Creating model: ('UVDoc', None)
Model files already exist. Using cached files. To redownload, please delete the directory manually: `/home/antoniotru/.paddlex/official_models/UVDoc`.
Creating model: ('PP-OCRv5_server_det', None)
Model files already exist. Using cached files. To redownload, please delete the directory manually: `/home/antoniotru/.paddlex/official_models/PP-OCRv5_server_det`.
Creating model: ('latin_PP-OCRv5_mobile_rec', None)
Model files already exist. Using cached files. To redownload, please delete the directory manually: `/home/antoniotru/.paddlex/official_models/latin_PP-OCRv5_mobile_rec`.
[ WARN:0@8.670] global cap_v4l.cpp:914 open VIDEOIO(V4L2:/dev/video2): can't open camera by index
[ERROR:0@8.784] global obsensor_uvc_stream_channel.cpp:163 getStreamChannelGroup Camera index out of range
Video source cannot be opened: 2
```

This means that the process did not found a functional video devide to use. So you must:
- Confirm that you have a functional connected webcam/camera.
- Verify the listed video output in terminal.
- Do camera tests in other applications to confirm webcam functionality.
- Ensure the correct listed video device in your scripts.
  
## Contributing
Contributions are appreciated. Please follow this steps to contribute:
1. Clon the repository.
2. Create a new branch.
3. Make your changes in the new branch.
4. Commit your changes.
5. Make a push inside your own branch.
6. Make a Pull Request.
