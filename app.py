"""Gradio demo launcher (Hugging Face Spaces entry point).

Local use:  pip install 'atlas-sonar[demo]' && python app.py
"""

from atlas.cli import build_demo

if __name__ == "__main__":
    build_demo().launch()
