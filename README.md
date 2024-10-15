# streamlit-blink-detection

**cv2.VideoCapture()**와 같은 비디오 캡처 및 GUI 함수는 Streamlit Cloud에서는 지원되지 않습니다. 대신 Streamlit 자체의 기능을 사용해야 합니다.  

**st.camera_input**을 사용하여 이미지를 캡처하고 처리하거나, 실시간 비디오 처리는 Streamlit Cloud에서는 구현하기 어려울 수 있습니다.
