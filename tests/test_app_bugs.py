import pytest
from unittest.mock import MagicMock, patch
import app
import os

def test_process_video_file_not_found(mocker):
    mock_cv2 = mocker.patch.object(app, 'cv2')
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_cv2.VideoCapture.return_value = mock_cap

    with pytest.raises(ValueError, match="Could not open video file or stream"):
        app.process_video('non_existent.mp4', 0.5, 0.5, 1.0, 'yolov8n.pt')

def test_process_video_releases_resources(mocker):
    mock_cv2 = mocker.patch.object(app, 'cv2')
    mock_cap = MagicMock()
    mock_cap.isOpened.side_effect = [True, False] # loop once then exit
    mock_cap.read.return_value = (True, MagicMock())
    mock_cap.get.return_value = 30 # fps, w, h
    mock_cv2.VideoCapture.return_value = mock_cap

    mock_out = MagicMock()
    mock_cv2.VideoWriter.return_value = mock_out

    # Mock model
    mock_model = MagicMock()
    app.model = mock_model
    app.current_model_name = 'yolov8n.pt'

    try:
        app.process_video('valid.mp4', 0.5, 0.5, 1.0, 'yolov8n.pt')
    except Exception:
        pass # We just care about release being called

    mock_cap.release.assert_called_once()
    mock_out.release.assert_called_once()

def test_realtime_detection_releases_resources(mocker):
    mock_cv2 = mocker.patch.object(app, 'cv2')
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None) # immediately exit loop
    mock_cv2.VideoCapture.return_value = mock_cap

    gen = app.realtime_detection(0.5, 0.5, 'yolov8n.pt', 'Webcam')
    try:
        next(gen)
    except StopIteration:
        pass

    mock_cap.release.assert_called_once()

def test_realtime_detection_screen_capture_close(mocker):
    mock_mss = mocker.patch.object(app, 'mss')
    mock_sct = MagicMock()
    mock_sct.monitors = [MagicMock()]
    mock_mss.mss.return_value = mock_sct

    mock_cv2 = mocker.patch.object(app, 'cv2')
    # Let model throw to break the while True loop
    app.model = MagicMock(side_effect=Exception("stop"))
    app.current_model_name = 'yolov8n.pt'

    gen = app.realtime_detection(0.5, 0.5, 'yolov8n.pt', 'Screen Capture')
    try:
        next(gen)
    except Exception as e:
        if str(e) != "stop":
            raise

    mock_sct.close.assert_called_once()

def test_process_youtube_video_cleanup(mocker):
    mock_yt_cls = mocker.patch.object(app, 'YouTube')
    mock_yt = MagicMock()
    mock_stream = MagicMock()
    mock_stream.download.return_value = '/tmp/fake_youtube_download.mp4'
    mock_yt.streams.filter.return_value.first.return_value = mock_stream
    mock_yt_cls.return_value = mock_yt

    mock_process_video = mocker.patch.object(app, 'process_video', return_value='/tmp/output.mp4')

    mock_os = mocker.patch.object(app, 'os')
    mock_os.path.exists.return_value = True

    app.process_youtube_video('http://youtube.com/fake', 0.5, 0.5, 1.0, 'yolov8n.pt')

    mock_os.remove.assert_called_once_with('/tmp/fake_youtube_download.mp4')
