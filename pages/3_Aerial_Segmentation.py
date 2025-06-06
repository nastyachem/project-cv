import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import colors
import io
from collections import OrderedDict
from utils import load_image_from_url, create_threshold_slider

st.set_page_config(page_title="Aerial Segmentation", page_icon="🛰️", layout="wide")

class UNet(nn.Module):
    """U-Net архитектура для семантической сегментации"""
    
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, 1)
        
    def conv_block(self, in_channels, out_channels):
        """Блок из двух конволюций с ReLU и BatchNorm"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)





@st.cache_resource
def load_unet_model():
    """Загружаем UNet модель для сегментации"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Попробуем несколько способов загрузки модели
        try:
            # Способ 1: torch.jit.load
            model = torch.jit.load('weights/aerial_forest_best.pt', map_location=device)
        except Exception as e1:
            try:
                # Способ 2: torch.load
                model = torch.load('weights/aerial_forest_best.pt', map_location=device)
            except Exception as e2:
                try:
                    # Способ 3: создаем UNet и загружаем веса
                    model = UNet(in_channels=3, out_channels=2)
                    checkpoint = torch.load('weights/aerial_forest_best.pt', map_location=device)
                    if 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                except Exception as e3:
                    st.error(f"Ошибка загрузки модели. Проверьте файл weights/aerial_forest_best.pt")
                    return None, None
        
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Критическая ошибка загрузки модели: {e}")
        return None, None

def preprocess_image(image, target_size=(512, 512)):
    """Предобработка изображения для UNet"""
    # Изменяем размер
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Преобразование в тензор
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image).unsqueeze(0)
    return tensor, image

def denormalize_image(tensor):
    """Денормализация изображения для визуализации"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Конвертируем в numpy
    image = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Денормализация
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    return image

def create_segmentation_overlay(original_image, segmentation_mask, alpha=0.6):
    """Создаем overlay сегментации на оригинальном изображении"""
    try:
        # Убеждаемся что маска имеет правильные размеры
        if segmentation_mask is None:
            st.error("❌ Маска сегментации пустая")
            return original_image
            
        # Цветовая карта для классов
        colormap = np.array([
            [0, 0, 0],        # Background (черный)
            [255, 255, 255],  # Forest (белый)
        ], dtype=np.uint8)
        
        # Проверяем диапазон значений в маске
        mask_min, mask_max = segmentation_mask.min(), segmentation_mask.max()
        if mask_max >= len(colormap):
            st.warning(f"⚠️ Значения в маске ({mask_min}-{mask_max}) превышают количество классов ({len(colormap)})")
            # Ограничиваем значения
            segmentation_mask = np.clip(segmentation_mask, 0, len(colormap)-1)
        
        # Создаем цветную маску
        colored_mask = colormap[segmentation_mask]
        
        # Конвертируем в numpy array
        original_array = np.array(original_image)
        
        # Убеждаемся что размеры совпадают
        if original_array.shape[:2] != colored_mask.shape[:2]:
            # Изменяем размер маски под оригинальное изображение
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray(colored_mask)
            mask_pil = mask_pil.resize((original_array.shape[1], original_array.shape[0]), PILImage.NEAREST)
            colored_mask = np.array(mask_pil)
        
        # Создаем overlay
        overlay = cv2.addWeighted(original_array, 1-alpha, colored_mask, alpha, 0)
        
        return Image.fromarray(overlay.astype(np.uint8))
        
    except Exception as e:
        st.error(f"❌ Ошибка создания overlay: {e}")
        return original_image

def visualize_segmentation_results(original_image, input_tensor, segmentation_mask, model_output):
    """Создаем детальную визуализацию результатов сегментации"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Оригинальное изображение
    axes[0].imshow(np.array(original_image))
    axes[0].set_title('Оригинальное изображение', fontsize=12)
    axes[0].axis('off')
    
    # 2. Маска сегментации
    cmap = colors.ListedColormap(['black', 'white'])
    im = axes[1].imshow(segmentation_mask, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title('Маска сегментации', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

def process_aerial_image(image, model, device, threshold=0.5):
    """Обрабатываем аэрокосмическое изображение"""
    try:
        # Проверяем что модель загружена
        if model is None:
            return None, None, {}, None, None
            
        # Предобработка
        input_tensor, resized_image = preprocess_image(image)
        input_tensor = input_tensor.to(device)
        
        # Инференс
        with torch.no_grad():
            output = model(input_tensor)
            
            # Обработка результатов с учетом разных форматов
            if len(output.shape) == 4:  # [batch, channels, height, width]
                if output.shape[1] == 1:
                    # Одноканальный выход (binary segmentation)
                    predictions = torch.sigmoid(output)
                    segmentation_mask = (predictions > threshold).float().cpu().numpy()[0, 0]
                elif output.shape[1] == 2:
                    # Двухканальный выход (background + forest)
                    predictions = torch.softmax(output, dim=1)
                    segmentation_mask = torch.argmax(predictions, dim=1).cpu().numpy()[0]
                else:
                    # Многоканальный выход
                    predictions = torch.softmax(output, dim=1)
                    segmentation_mask = torch.argmax(predictions, dim=1).cpu().numpy()[0]
            else:
                return None, None, {}, output, input_tensor
            
            # Приводим к целочисленному типу
            segmentation_mask = segmentation_mask.astype(np.uint8)
            
    except Exception as e:
        st.error(f"❌ Ошибка в process_aerial_image: {e}")
        return None, None, {}, None, None
    
    # Создаем overlay
    overlay_image = create_segmentation_overlay(resized_image, segmentation_mask)
    
    # Статистика сегментации
    unique, counts = np.unique(segmentation_mask, return_counts=True)
    total_pixels = segmentation_mask.size
    
    stats = {}
    class_names = ['Background', 'Forest']
    for i, (class_id, count) in enumerate(zip(unique, counts)):
        if class_id < len(class_names):
            percentage = (count / total_pixels) * 100
            stats[class_names[class_id]] = {
                'pixels': int(count),
                'percentage': round(percentage, 2)
            }
    
    return overlay_image, segmentation_mask, stats, output, input_tensor

def create_segmentation_plot(segmentation_mask):
    """Создаем график сегментации"""
    try:
        if segmentation_mask is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, 'Ошибка сегментации', ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
            
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Определяем количество уникальных классов
        unique_classes = np.unique(segmentation_mask)
        max_class = int(segmentation_mask.max())
        
        # Цветовая карта в зависимости от количества классов
        if max_class <= 1:
            cmap = colors.ListedColormap(['black', 'white'])
            labels = ['Background', 'Forest']
        else:
            # Для большего количества классов
            cmap = plt.cm.get_cmap('tab10', max_class + 1)
            labels = [f'Class {i}' for i in range(max_class + 1)]
        
        im = ax.imshow(segmentation_mask, cmap=cmap, vmin=0, vmax=max_class)
        ax.set_title('Карта сегментации')
        ax.axis('off')
        
        # Добавляем colorbar
        if max_class <= 10:  # Только если классов не очень много
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks(list(range(max_class + 1)))
            cbar.set_ticklabels(labels[:max_class + 1])
        
        return fig
        
    except Exception as e:
        # В случае ошибки возвращаем пустой график с сообщением
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f'Ошибка: {str(e)}', ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig

def main():
    st.title("🛰️ Семантическая сегментация аэрокосмических снимков")
    

    
    # Характеристики
    with st.expander("ℹ️ Характеристики модели"):
        st.image("graphics/forest.jpeg", use_container_width=True)
        st.markdown("""
        
        - Архитектура: U-Net
        - Размер модели: 118MB
        - Классы: Background, Forest
        - Задача: Семантическая сегментация
        - Размер входа: 512x512
        """)
    
    # Загрузка модели
    model, device = load_unet_model()
    if model is None:
        st.error("❌ Модель не загружена. Проверьте путь к файлу weights/aerial_forest_best.pt")
        return
    
    # Настройки
    st.sidebar.header("⚙️ Настройки")
    threshold = create_threshold_slider(
        "Порог сегментации", 
        default_value=0.55,
        help_text="Порог для бинаризации маски сегментации"
    )
    
    # Фиксированные настройки
    show_detailed_view = False
    overlay_alpha = 0.6
    
    # Загрузка файлов
    st.header("📁 Загрузка аэрокосмических снимков")
    
    # Табы для разных способов загрузки
    tab1, tab2 = st.tabs(["📂 Загрузить файлы", "🌐 Загрузить по URL"])
    
    with tab1:
        uploaded_files = st.file_uploader(
            "Выберите спутниковые изображения",
            type=['jpg', 'jpeg', 'png', 'tiff'],
            accept_multiple_files=True,
            help="Поддерживаются форматы: JPG, JPEG, PNG, TIFF"
        )
    
    with tab2:
        url_input = st.text_input(
            "Введите URL спутникового изображения:",
            placeholder="https://example.com/satellite_image.jpg",
            help="Введите прямую ссылку на аэрокосмическое изображение"
        )
        
        url_files = None
        if url_input:
            with st.spinner("Загрузка спутникового изображения по URL..."):
                image, error = load_image_from_url(url_input)
                if image:
                    # Создаем псевдо-файл для совместимости с основным кодом
                    url_files = [type('MockFile', (), {
                        'name': url_input.split('/')[-1] or 'url_satellite_image.jpg'
                    })]
                    # Сохраняем изображение в session_state
                    st.session_state.url_satellite_image = image
                    st.success("✅ Спутниковое изображение загружено успешно!")
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
            st.subheader(f"🛰️ Снимок {idx + 1}: {uploaded_file.name}")
            
            try:
                # Загружаем изображение
                if hasattr(uploaded_file, 'read'):
                    # Обычный загруженный файл
                    image = Image.open(uploaded_file).convert('RGB')
                else:
                    # URL-изображение из session_state
                    image = st.session_state.url_satellite_image
                
                # Обработка
                with st.spinner("Выполнение сегментации..."):
                    overlay_image, segmentation_mask, stats, model_output, input_tensor = process_aerial_image(
                        image, model, device, threshold
                    )
                
                # Проверяем что сегментация прошла успешно
                if overlay_image is None or segmentation_mask is None:
                    st.error(f"❌ Не удалось выполнить сегментацию для {uploaded_file.name}")
                    continue
                
                if show_detailed_view:
                    # Детальная визуализация
                    st.markdown("**🔍 Детальный анализ сегментации:**")
                    detailed_fig = visualize_segmentation_results(
                        image, input_tensor, segmentation_mask, model_output
                    )
                    st.pyplot(detailed_fig)
                    plt.close(detailed_fig)
                else:
                    # Стандартная визуализация
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Оригинал:**")
                        st.image(image, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Карта классов:**")
                        seg_fig = create_segmentation_plot(segmentation_mask)
                        st.pyplot(seg_fig)
                        plt.close(seg_fig)
                

                
                # Кнопки скачивания
                col_download1, col_download2 = st.columns(2)
                
                with col_download1:
                    # Скачивание overlay
                    img_buffer = io.BytesIO()
                    overlay_image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="💾 Скачать сегментацию",
                        data=img_buffer.getvalue(),
                        file_name=f"segmented_{uploaded_file.name}",
                        mime="image/png"
                    )
                
                with col_download2:
                    # Скачивание детального анализа
                    if show_detailed_view:
                        detailed_buffer = io.BytesIO()
                        detailed_fig.savefig(detailed_buffer, format='png', dpi=150, bbox_inches='tight')
                        detailed_buffer.seek(0)
                        
                        st.download_button(
                            label="📊 Скачать анализ",
                            data=detailed_buffer.getvalue(),
                            file_name=f"analysis_{uploaded_file.name.split('.')[0]}.png",
                            mime="image/png"
                        )
                
                st.divider()
                
            except Exception as e:
                st.error(f"Ошибка обработки {uploaded_file.name}: {e}")
    
    else:
        pass

if __name__ == "__main__":
    main() 