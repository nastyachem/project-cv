from PIL import Image
import io
import requests
import streamlit as st

def create_threshold_slider(label, default_value, help_text=None):
    """Ползунок порога"""
    return st.sidebar.slider(
        label,
        min_value=0.1,
        max_value=1.0,
        value=default_value,
        step=0.05,
        help=help_text
    )

def load_image_from_url(url):
    """Загрузка изображения по URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        return image, None
    except requests.exceptions.RequestException as e:
        return None, f"Ошибка загрузки: {e}"
    except Exception as e:
        return None, f"Ошибка обработки изображения: {e}" 