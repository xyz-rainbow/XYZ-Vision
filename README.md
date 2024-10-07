# XYZ Vision - Gradio WebUI for YOLO (ultralytics)

   This project provides a user interface for object detection using YOLO models. 

## Input methods including:
   
   - Image upload
   - Video upload
   - Real-time Webcam
   - Screen capture
   - YouTube video processing. # Not Avaiable

## Features

   - **Real-time object detection** from webcam or screen capture
   - It can work with **GPU** (Prefered) And it supports **"CPU Only"** if needed.
   - Support for **multiple YOLO models** *(nano, small, medium, large, xlarge)*
   - **Image object detection** with confidence and IoU thresholds
   - **Video object detection** with adjustable processing speed
   - *YouTube video processing* **#NOTWORKINGFORNOW**
   - **Text description of detected objects** # Maybe not working

## **Installation**

1. Clone this repository:
   ```
   git clone https://github.com/rainbowkode/XYZ-Vision
   cd XYZ-Vision
   ```

2. Install the required dependencies and requirements:
   ```
   pip install -r requirements.txt # Prefered
      # OR
   python install-dependencies.py # if not workin, try "python3" or "py"
   ```

## **Usage**

1. Run the application:
   ```
   python app.py
      #OR
   ./app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:7860`).

3. Use the interface and select your desired input method

4. Adjust the detection parameters.

6. Click the corresponding button to start object detection.

## *Notes*

- Ensure you have a CUDA-capable **GPU for faster processing** (optional but recommended)
- The application will download the selected model if it's not already present in the working directory.
- #NOTWORKINGFORNOW - For YouTube video processing, ensure you have a stable internet connection.

## Future Improvements

- Add object tracking and trajectory analysis
- Implement 3D coordinate estimation
- Add velocity and acceleration calculations for detected objects
- Develop a separate interface for model tuning and dataset selection
- Implement stop, pause, and play controls for video processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License [FOR NOW]

For the benefit of the public, this project is open source and available under the unlicense terms.
This License can change at any moment, please check this on future versions of XYZ Vision.
