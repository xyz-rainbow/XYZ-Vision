import pytest
import importlib
import torch

def test_device_selection_mps(mocker):
    # Mock torch.cuda.is_available = False
    mocker.patch('torch.cuda.is_available', return_value=False)
    # Mock torch.backends.mps.is_available = True
    mock_mps = mocker.MagicMock()
    mock_mps.is_available.return_value = True
    mocker.patch('torch.backends.mps', mock_mps, create=True)

    # We need to reload app.py to trigger device selection logic again
    import app
    importlib.reload(app)

    assert app.device == 'mps'

def test_device_selection_cpu(mocker):
    mocker.patch('torch.cuda.is_available', return_value=False)
    if hasattr(torch.backends, 'mps'):
        mocker.patch('torch.backends.mps.is_available', return_value=False)

    import app
    importlib.reload(app)
    assert app.device == 'cpu'

def test_model_reloading(mocker):
    import app
    mocker.patch('app.YOLO')
    app.current_model_name = None
    app.model = None

    # Initial load
    app.load_model('yolov8n.pt')
    assert app.current_model_name == 'yolov8n.pt'
    first_model = app.model
    app.YOLO.assert_called_once_with('yolov8n.pt')

    # Second load with same model shouldn't recreate in process_image
    app.YOLO.reset_mock()
    # mock input image
    mock_img = mocker.MagicMock()
    # mock model inference
    app.model.return_value = [mocker.MagicMock()]
    app.model.return_value[0].names = ['person']
    app.model.return_value[0].boxes = []

    app.process_image(mock_img, 0.5, 0.5, 'yolov8n.pt')
    app.YOLO.assert_not_called()
    assert app.current_model_name == 'yolov8n.pt'

    # Third load with different model should recreate
    app.process_image(mock_img, 0.5, 0.5, 'yolov8s.pt')
    app.YOLO.assert_called_once_with('yolov8s.pt')
    assert app.current_model_name == 'yolov8s.pt'

def test_process_youtube_video_no_stream(mocker):
    import app
    mock_yt = mocker.MagicMock()
    # Mock first() to return None
    mock_yt.streams.filter().first.return_value = None
    mocker.patch('app.YouTube', return_value=mock_yt)

    with pytest.raises(ValueError, match="No suitable mp4 stream found for this YouTube video"):
        app.process_youtube_video('http://youtube.com/fake', 0.5, 0.5, 1.0, 'yolov8n.pt')

def test_generate_description(mocker):
    import app
    mock_result = mocker.MagicMock()
    mock_result.names = {0: 'person', 1: 'car'}

    box1 = mocker.MagicMock()
    box1.cls = 0
    box1.conf = 0.95

    box2 = mocker.MagicMock()
    box2.cls = 1
    box2.conf = 0.82

    mock_result.boxes = [box1, box2]

    desc = app.generate_description(mock_result)
    assert "person (Confidence: 0.95)" in desc
    assert "car (Confidence: 0.82)" in desc
