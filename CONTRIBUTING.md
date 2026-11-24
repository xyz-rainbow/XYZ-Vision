**English** | [Español](CONTRIBUTING.es.md)

# Contributing to XYZ-Vision

Thanks for taking the time to contribute. Gradio WebUI for YOLO (Ultralytics) object detection on image, video, webcam, and screen, with GPU or CPU.

## License and contributions

By opening an issue or sending a pull request you accept the contribution clause in [LICENSE](LICENSE): you assign to Rainbow Technology all rights in your contribution, or, if that assignment is not effective, you grant a perpetual, irrevocable, exclusive license, and you warrant that you have the right to contribute it.

Please do not open a public issue for a security vulnerability; use the repository Security advisory form if it is available.

## Development setup

Follow [README.md](README.md):

```bash
git clone https://github.com/xyz-rainbow/XYZ-Vision
cd XYZ-Vision
pip install -r requirements.txt
python app.py
```

CI is `.github/workflows/ci.yml`.

## Pull requests

1. Create a topic branch from the default branch (`main`).
2. Keep the change focused.
3. Update [CHANGELOG.md](CHANGELOG.md) and add a line to `docs/logs/YYYY-MM.md` when the change is user-visible or touches process or docs.
4. Never commit secrets, API keys, or private data.
5. Wait for CI to go green when the repository has workflows. Maintainers squash-merge into the default branch.
