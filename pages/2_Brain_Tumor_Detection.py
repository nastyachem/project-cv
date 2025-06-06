import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
import torch
import io
from utils import load_image_from_url, create_threshold_slider
from urllib.parse import urlparse

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

@st.cache_resource
def load_brain_model(model_type):
    """Загружаем модель детекции опухолей мозга"""
    try:
        model_path = f'weights/{model_type}.pt'
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели {model_type}: {e}")
        return None





def draw_detections(image, results, confidence_threshold):
    """Рисуем bounding boxes на изображении"""
    image_array = np.array(image)
    
    detections = []
    for box in results[0].boxes:
        if box.conf >= confidence_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf)
            
            # Рисуем bounding box
            cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Добавляем текст с уверенностью
            label = f"Tumor: {confidence:.3f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(image_array, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image_array, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'area': (x2 - x1) * (y2 - y1)
            })
    
    return Image.fromarray(image_array), detections

def process_brain_image(image, model, confidence_threshold):
    """Обрабатываем МРТ изображение"""
    results = model(image, conf=confidence_threshold)
    
    if len(results[0].boxes) > 0:
        result_image, detections = draw_detections(image, results, confidence_threshold)
        detection_info = {
            'tumors_count': len(detections),
            'detections': detections
        }
        return result_image, detection_info, True
    else:
        return image, {'tumors_count': 0, 'detections': []}, False

def main():
    st.title("🧠 Детекция опухолей мозга")
    
    # Характеристики
    with st.expander("ℹ️ Характеристики модели"):
        st.image("graphics/brain.jpeg", use_container_width=True)
        st.markdown("""
        
        - Архитектура: YOLOv12m
        - Размер модели: 38.9MB
        - Доступные плоскости: Axial, Coronal, Sagittal, Brain All
        - Количество классов: 2 (positive,negative)
        - Задача: Детекция объектов
        """)
    
    # Настройки
    st.sidebar.header("⚙️ Настройки")
    
    # Выбор модели
    model_type = st.sidebar.selectbox(
        "Выберите плоскость МРТ:",
        ["axial", "coronal", "sagittal", "brain_all"],
        index=0,
        help="Axial - поперечные срезы, Coronal - фронтальные, Sagittal - боковые, Brain All - универсальная модель"
    )
    
    confidence_threshold = create_threshold_slider(
        "Порог уверенности", 
        default_value=0.4,
        help_text="Минимальная уверенность для детекции опухоли"
    )
    
    # Загрузка модели
    model = load_brain_model(model_type)
    
    # Загрузка файлов
    st.header("📁 Загрузка МРТ изображений")
    
    # Табы для разных способов загрузки
    tab1, tab2 = st.tabs(["📂 Загрузить файлы", "🌐 Загрузить по URL"])
    
    with tab1:
        uploaded_files = st.file_uploader(
            "Выберите МРТ изображения",
            type=['jpg', 'jpeg', 'png', 'tiff'],
            accept_multiple_files=True,
            help="Поддерживаются форматы: JPG, JPEG, PNG, TIFF"
        )
    
    with tab2:
        url_input = st.text_input(
            "Введите URL МРТ изображения:",
            placeholder="https://example.com/brain_scan.jpg",
            help="Введите прямую ссылку на МРТ изображение"
        )
        
        url_files = None
        if url_input:
            with st.spinner("Загрузка МРТ изображения по URL..."):
                image, error = load_image_from_url(url_input)
                if image:
                    # Создаем псевдо-файл для совместимости с основным кодом
                    url_files = [type('MockFile', (), {
                        'name': url_input.split('/')[-1] or 'url_mri_image.jpg'
                    })]
                    # Сохраняем изображение в session_state
                    st.session_state.url_mri_image = image
                    st.success("✅ МРТ изображение загружено успешно!")
                else:
                    st.error(f"❌ {error}")
    
    # Объединяем файлы из обоих источников
    if uploaded_files or url_files:
        if uploaded_files and url_files:
            all_files = uploaded_files + url_files
        elif uploaded_files:
            all_files = uploaded_files
        else:
            all_files = url_files
        uploaded_files = all_files
    
    # Обработка изображений
    if uploaded_files:
        st.success(f"Загружено {len(uploaded_files)} файл(ов)")
        
        for idx, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"🧠 МРТ {idx + 1}: {uploaded_file.name}")
            st.caption(f"Анализ с помощью модели: {model_type.upper()}")
            
            try:
                # Загружаем изображение
                if hasattr(uploaded_file, 'read'):
                    # Обычный загруженный файл
                    image = Image.open(uploaded_file).convert('RGB')
                else:
                    # Изображение из URL (из session_state)
                    image = st.session_state.url_mri_image
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Оригинал:**")
                    st.image(image, use_container_width=True)
                
                # Обработка
                with st.spinner("Анализ МРТ изображения..."):
                    result_image, detection_info, tumors_found = process_brain_image(
                        image, model, confidence_threshold
                    )
                
                with col2:
                    st.markdown("**Результат детекции:**")
                    st.image(result_image, use_container_width=True)
                
                # Статистика
                st.markdown("**📊 Результаты анализа:**")
                if tumors_found:
                    st.warning(f"🚨 Обнаружено потенциальных опухолей: {detection_info['tumors_count']}")
                else:
                    st.success("✅ Опухоли не обнаружены")
                
                # Кнопка скачивания
                img_buffer = io.BytesIO()
                result_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label="💾 Скачать результат анализа",
                    data=img_buffer.getvalue(),
                    file_name=f"analyzed_{uploaded_file.name}",
                    mime="image/png"
                )
                
                st.divider()
                
            except Exception as e:
                st.error(f"Ошибка обработки {uploaded_file.name}: {e}")
    
    else:
        pass

if __name__ == "__main__":
    main() 