import streamlit as st

# Настройка страницы
st.set_page_config(
    page_title="CV Detection Service",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Главная страница
def main():
    st.title("🤖 CV для ваших изображений")
    st.markdown("""
    
    
     ** ⬅️ Выберите подходящую модель под вашу задачу в боковом меню ⬅️ **
     
                ✔ Каждая модель поддерживает загрузку файлов и прямые ссылки для загрузки
    ### 👤 Детекция лиц и их маскировка
    - Обнаружение лиц на изображениях с помощью YOLO12
    - Скройте свое лицо с помощью блюра
    
    
    ### 🧠 Детекция опухолей мозга  
    - Обнаружение опухолей на МРТ снимках с помощью YOLO12
    - Анализ аксиальных T1WCE изображений
    
    ### 🛰️ Сегментация аэрокосмических снимков
    - Семантическая сегментация с помощью UNet
    - Обработка спутниковых изображений и выделение лесных массивов
    
    
    ---

    """)

if __name__ == "__main__":
    main() 