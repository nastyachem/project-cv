import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
import torch
import io
from utils import load_image_from_url, create_threshold_slider

st.set_page_config(page_title="Face Detection", page_icon="👤", layout="wide")

@st.cache_resource
def load_face_model():
    """Загрузка модели"""
    try:
        model = YOLO('weights/face.pt')
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

def fix_image_orientation(image):
    """Исправляем ориентацию изображения на основе EXIF данных"""
    try:
        # Автоматически исправляем ориентацию изображения
        image = ImageOps.exif_transpose(image)
        return image
    except Exception:
        # Если возникла ошибка, возвращаем исходное изображение
        return image

def apply_face_mask(image, boxes, confidence_threshold=0.5):
    """Применяем маску блюра на обнаруженные лица"""
    image_array = np.array(image)
    
    for box in boxes:
        if box.conf >= confidence_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Создаем размытие для маскировки
            face_region = image_array[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 40)
            
            # Применяем маску
            image_array[y1:y2, x1:x2] = blurred_face
    
    return Image.fromarray(image_array)

def process_image(image, model, confidence_threshold):
    """Обрабатываем изображение"""
    results = model(image, conf=confidence_threshold)
    
    if len(results[0].boxes) > 0:
        masked_image = apply_face_mask(image, results[0].boxes, confidence_threshold)
        detection_info = {
            'faces_count': len(results[0].boxes),
            'confidences': [float(box.conf) for box in results[0].boxes]
        }
        return masked_image, detection_info, True
    else:
        return image, {'faces_count': 0, 'confidences': []}, False

def main():
    st.title("👤 Детекция и маскировка лиц")
    
    # Характеристики
    with st.expander("ℹ️ Характеристики модели"):
        st.image("graphics/face.png", use_container_width=True)
        st.markdown("""
        
        - Архитектура: YOLOv12n
        - Размер модели: 5.2MB
        - Количество классов: 1 (Human Face)
        - Объем выборки: 12,880 изображений
        - Задача: Детекция объектов
        """)
    
    # Загрузка модели
    model = load_face_model()
    if model is None:
        st.error("Модель не загружена. Проверьте путь к файлу весов.")
        return
    
    # Настройки
    st.sidebar.header("⚙️ Настройки")
    confidence_threshold = create_threshold_slider(
        "Порог уверенности", 
        default_value=0.5,
        help_text="Минимальная уверенность для детекции лица"
    )
    
    # Загрузка файлов
    st.header("📁 Загрузка изображений")
    
    # Табы для разных способов загрузки
    tab1, tab2 = st.tabs(["📂 Загрузить файлы", "🌐 Загрузить по URL"])
    
    with tab1:
        uploaded_files = st.file_uploader(
            "Выберите изображения",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Поддерживаются форматы: JPG, JPEG, PNG"
        )
    
    with tab2:
        url_input = st.text_input(
            "Введите URL изображения:",
            placeholder="https://example.com/image.jpg",
            help="Введите прямую ссылку на изображение"
        )
        
        url_files = None
        if url_input:
            with st.spinner("Загрузка изображения по URL..."):
                image, error = load_image_from_url(url_input)
                if image:
                    # Создаем псевдо-файл для совместимости с основным кодом
                    url_files = [type('MockFile', (), {
                        'name': url_input.split('/')[-1] or 'url_image.jpg'
                    })]
                    # Сохраняем изображение в session_state
                    st.session_state.url_image = image
                    st.success("✅ Изображение загружено успешно!")
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
    
    if uploaded_files:
        st.success(f"Загружено {len(uploaded_files)} файл(ов)")
        
        # Обработка изображений
        for idx, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"🖼️ Изображение {idx + 1}: {uploaded_file.name}")
            
            try:
                # Загружаем изображение и исправляем ориентацию
                if hasattr(uploaded_file, 'read'):
                    # Обычный загруженный файл
                    image = Image.open(uploaded_file).convert('RGB')
                    image = fix_image_orientation(image)
                else:
                    # Изображение из URL (из session_state)
                    image = st.session_state.url_image
                    image = fix_image_orientation(image)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Оригинал:**")
                    st.image(image, use_container_width=True)
                
                # Обработка
                with st.spinner("Обработка изображения..."):
                    masked_image, detection_info, faces_found = process_image(
                        image, model, confidence_threshold
                    )
                
                with col2:
                    st.markdown("**Результат:**")
                    st.image(masked_image, use_container_width=True)
                
                # Статистика
                st.markdown("**📊 Результаты детекции:**")
                if faces_found:
                    st.success(f"Обнаружено лиц: {detection_info['faces_count']}")
                    if detection_info['confidences']:
                        avg_conf = np.mean(detection_info['confidences'])
                        st.info(f"Средняя уверенность: {avg_conf:.3f}")
                else:
                    st.warning("Лица не обнаружены")
                
                # Кнопка скачивания
                img_buffer = io.BytesIO()
                masked_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label="💾 Скачать обработанное изображение",
                    data=img_buffer.getvalue(),
                    file_name=f"masked_{uploaded_file.name}",
                    mime="image/png"
                )
                
                st.divider()
                
            except Exception as e:
                st.error(f"Ошибка обработки {uploaded_file.name}: {e}")
    
    else:
        pass

if __name__ == "__main__":
    main() 