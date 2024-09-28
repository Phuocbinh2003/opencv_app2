import streamlit as st

# Khởi tạo session state cho tọa độ chuột
if 'mouse_x' not in st.session_state:
    st.session_state.mouse_x = 0
if 'mouse_y' not in st.session_state:
    st.session_state.mouse_y = 0

# Tạo một placeholder để hiển thị vị trí chuột
mouse_pos_placeholder = st.empty()

# JavaScript để theo dõi vị trí chuột
st.markdown("""
    <script>
    function updateMousePosition(event) {
        const x = event.clientX;
        const y = event.clientY;
        
        // Gửi vị trí chuột về Streamlit
        const data = {x: x, y: y};
        window.parent.streamlit.setMousePosition(data);
    }

    document.addEventListener('mousemove', updateMousePosition);
    </script>
""", unsafe_allow_html=True)

# Hàm cập nhật vị trí chuột
def update_mouse_position():
    st.session_state.mouse_x = st.session_state.mouse_position['x']
    st.session_state.mouse_y = st.session_state.mouse_position['y']
    mouse_pos_placeholder.write(f"Vị trí chuột: (X: {st.session_state.mouse_x}, Y: {st.session_state.mouse_y})")

# Thiết lập hàm callback để gọi khi có sự thay đổi
if 'mouse_position' not in st.session_state:
    st.session_state.mouse_position = {'x': 0, 'y': 0}

# Nút để lấy vị trí chuột
if st.button('Lấy vị trí chuột'):
    update_mouse_position()

# Hiển thị vị trí chuột ban đầu
mouse_pos_placeholder.write(f"Vị trí chuột: (X: {st.session_state.mouse_x}, Y: {st.session_state.mouse_y})")
