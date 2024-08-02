# YOLOv8 Object Detection UI

This project provides a user interface for YOLOv8 object detection using Gradio. It supports various input methods including image upload, video upload, webcam, screen capture, and YouTube video processing.

## Features

- Support for multiple YOLOv8 models (nano, small, medium, large, xlarge)
- Image object detection with confidence and IoU thresholds
- Video object detection with adjustable processing speed
- Real-time object detection from webcam or screen capture
- YouTube video processing
- Text description of detected objects

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/yolov8-object-detection-ui.git
   cd yolov8-object-detection-ui
   ```

2. Install the required dependencies:
   ```
   python install-dependencies.py
   ```

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:7860`).

3. Use the interface to select the desired input method and adjust detection parameters.

4. Click the corresponding button to start object detection.

## Notes

- Ensure you have a CUDA-capable GPU for faster processing (optional but recommended).
- The application will download the selected YOLOv8 model if it's not already present in the working directory.
- For YouTube video processing, ensure you have a stable internet connection.

## Future Improvements

- Add object tracking and trajectory analysis
- Implement 3D coordinate estimation
- Add velocity and acceleration calculations for detected objects
- Develop a separate interface for model tuning and dataset selection
- Implement stop, pause, and play controls for video processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
