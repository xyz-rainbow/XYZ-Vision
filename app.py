import gradio as gr
from ultralytics import YOLO
import cv2
import torch
import numpy as np
import tempfile
import time
import os
import json
from pytube import YouTube
from PIL import Image
import mss

# Ensure GPU usage if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize YOLOv8 model
model = None
available_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']

def load_model(model_name):
    global model
    model = YOLO(model_name).to(device)

load_model(available_models[0])  # Load default model

def process_image(input_image, conf_threshold, iou_threshold, selected_model):
    global model
    if model is None or model.model.names[0] != selected_model:
        load_model(selected_model)
    
    results = model(input_image, conf=conf_threshold, iou=iou_threshold)
    annotated_image = results[0].plot()
    
    # Generate text description
    description = generate_description(results[0])
    
    return annotated_image, description

def process_video(input_video, conf_threshold, iou_threshold, speed_factor, selected_model):
    global model
    if model is None or model.model.names[0] != selected_model:
        load_model(selected_model)
    
    output_path = tempfile.mktemp('.mp4')
    
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    batch_size = 4
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % max(1, int(1/speed_factor)) != 0:
            continue
        
        frames.append(frame)
        
        if len(frames) == batch_size:
            results = model(frames, conf=conf_threshold, iou=iou_threshold)
            for r in results:
                out.write(r.plot())
            frames = []

        time.sleep(max(0, (1/fps/speed_factor) - (time.time() - time.perf_counter())))

    if frames:
        results = model(frames, conf=conf_threshold, iou=iou_threshold)
        for r in results:
            out.write(r.plot())

    cap.release()
    out.release()
    
    return output_path

def realtime_detection(conf_threshold, iou_threshold, selected_model, input_source):
    global model
    if model is None or model.model.names[0] != selected_model:
        load_model(selected_model)
    
    if input_source == "Webcam":
        cap = cv2.VideoCapture(0)
    elif input_source == "Screen Capture":
        sct = mss.mss()
        monitor = sct.monitors[0]
    else:
        raise ValueError("Invalid input source")

    while True:
        if input_source == "Webcam":
            ret, frame = cap.read()
            if not ret:
                break
        elif input_source == "Screen Capture":
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        results = model(frame, conf=conf_threshold, iou=iou_threshold)
        yield results[0].plot()

def generate_description(result):
    classes = result.names
    boxes = result.boxes
    
    description = "Detected objects:\n"
    for box in boxes:
        class_id = int(box.cls)
        conf = float(box.conf)
        description += f"- {classes[class_id]} (Confidence: {conf:.2f})\n"
    
    return description

def process_youtube_video(youtube_url, conf_threshold, iou_threshold, speed_factor, selected_model):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    video_path = stream.download(output_path=tempfile.gettempdir())
    
    return process_video(video_path, conf_threshold, iou_threshold, speed_factor, selected_model)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# YOLOv8 Object Detection")
    
    model_dropdown = gr.Dropdown(choices=available_models, value=available_models[0], label="Select Model")
    
    with gr.Tab("Image"):
        with gr.Row():
            image_input = gr.Image(type="numpy")
            image_output = gr.Image(type="numpy")
        
        with gr.Row():
            conf_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold")
            iou_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.45, step=0.05, label="IOU Threshold")
        
        image_button = gr.Button("Detect Objects")
        image_description = gr.Textbox(label="Detection Description")
    
    with gr.Tab("Video"):
        with gr.Row():
            video_input = gr.Video()
            video_output = gr.Video(label="Processed Video")
        
        with gr.Row():
            video_conf_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold")
            video_iou_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.45, step=0.05, label="IOU Threshold")
            speed_slider = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Processing Speed Factor")
        
        video_button = gr.Button("Detect Objects in Video")

    with gr.Tab("Real-time"):
        with gr.Row():
            realtime_output = gr.Image(type="numpy")
        
        with gr.Row():
            realtime_conf_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold")
            realtime_iou_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.45, step=0.05, label="IOU Threshold")
        
        input_source = gr.Radio(["Webcam", "Screen Capture"], label="Input Source", value="Webcam")
        realtime_button = gr.Button("Start Real-time Detection")

    with gr.Tab("YouTube Video"):
        youtube_url = gr.Textbox(label="YouTube Video URL")
        youtube_output = gr.Video(label="Processed YouTube Video")
        youtube_button = gr.Button("Process YouTube Video")

    image_button.click(process_image, inputs=[image_input, conf_slider, iou_slider, model_dropdown], outputs=[image_output, image_description])
    video_button.click(process_video, inputs=[video_input, video_conf_slider, video_iou_slider, speed_slider, model_dropdown], outputs=video_output)
    realtime_button.click(realtime_detection, inputs=[realtime_conf_slider, realtime_iou_slider, model_dropdown, input_source], outputs=realtime_output)
    youtube_button.click(process_youtube_video, inputs=[youtube_url, video_conf_slider, video_iou_slider, speed_slider, model_dropdown], outputs=youtube_output)

demo.launch()
