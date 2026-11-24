[English](CONTRIBUTING.md) | **Español**

# Contribuir a XYZ-Vision

Gracias por tomarte el tiempo de contribuir. Gradio WebUI for YOLO (Ultralytics) object detection on image, video, webcam, and screen, with GPU or CPU.

## Licencia y contribuciones

Al abrir un issue o enviar una pull request aceptas la cláusula de contribución de [LICENSE](LICENSE): cedes a Rainbow Technology todos los derechos sobre lo que aportas o, si esa cesión no fuera eficaz, le concedes una licencia perpetua, irrevocable y exclusiva, y garantizas que tienes derecho a aportarlo.

No abras un issue público para una vulnerabilidad; usa el formulario de avisos de seguridad del repositorio si está disponible.

## Entorno de desarrollo

Sigue [README.md](README.md):

```bash
git clone https://github.com/xyz-rainbow/XYZ-Vision
cd XYZ-Vision
pip install -r requirements.txt
python app.py
```

El CI está en `.github/workflows/ci.yml`.

## Pull requests

1. Crea una rama de trabajo desde la rama por defecto (`main`).
2. Mantén el cambio acotado.
3. Actualiza [CHANGELOG.md](CHANGELOG.md) y añade una línea en `docs/logs/YYYY-MM.md` cuando el cambio sea visible o toque el proceso o la documentación.
4. No subas secretos, claves de API ni datos privados.
5. Espera a que el CI esté en verde si el repositorio tiene workflows. Los responsables fusionan con squash en la rama por defecto.
