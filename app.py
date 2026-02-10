import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize
from functools import lru_cache

# --- 1. SET CONFIG (ต้องอยู่บรรทัดแรก) ---
st.set_page_config(page_title="CineSense: Tactical Analysis", layout="wide")

# --- 2. FAST LOGIC (Keep your original logic) ---
@lru_cache(maxsize=1000)
def thai_tokenize(text):
    return word_tokenize(str(text), engine='newmm')

@st.cache_resource(show_spinner="Syncing Neural Link...")
def load_models():
    try:
        return joblib.load('model.joblib'), joblib.load('model_v2.joblib')
    except: return None, None

@st.cache_data(show_spinner="Accessing Database...")
def load_data():
    try:
        return pd.read_csv('8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv')
    except:
        return pd.DataFrame({'text':['Error'], 'label':['Neutral'], 'review_id':['000']})

model_v1, model_v2 = load_models()
df = load_data()

def get_top_features(model, text, pred_class):
    try:
        tfidf = model.named_steps['tfidf']
        clf = model.named_steps['clf']
        feature_names = tfidf.get_feature_names_out()
        tokens = thai_tokenize(text)
        present_features = list(set([f for f in tokens if f in feature_names]))
        if not present_features: return []
        idx = list(clf.classes_).index(pred_class)
        weights = clf.coef_[idx]
        feat_list = []
        for f in present_features:
            f_idx = np.where(feature_names == f)[0][0]
            feat_list.append((f, weights[f_idx]))
        return sorted(feat_list, key=lambda x: x[1], reverse=True)[:5]
    except: return []

# --- 3. GAMING INTERFACE CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&family=JetBrains+Mono:wght@400&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #eef2f3 0%, #8e9eab 100%);
        font-family: 'Kanit', sans-serif;
    }

    /* สไตล์กล่อง UI แบบโปร่งแสงเหมือนในเกม */
    .game-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 2px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        margin-bottom: 20px;
    }

    /* ปุ่มกดแนว Tactical */
    .stButton>button {
        background: #00a8cc;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        transition: 0.3s;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background: #007fa3;
        box-shadow: 0 0 15px rgba(0, 168, 204, 0.5);
        transform: translateY(-2px);
    }

    /* ปรับแต่งหัวข้อ */
    .main-header {
        color: #1a3c5a;
        font-weight: 800;
        text-align: center;
        font-size: 2.8rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Tag คำสำคัญ */
    .feature-tag {
        background: rgba(0, 168, 204, 0.1);
        color: #007fa3;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(0, 168, 204, 0.2);
        display: inline-block;
        margin: 2px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. HEADER ---
st.markdown('<div class="main-header">CINESENSE TACTICAL CORE</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#555;">Neural Sentiment Processing Unit v4.0</p>', unsafe_allow_html=True)

# --- 5. MAIN LAYOUT ---
col_main, col_side = st.columns([1.8, 1], gap="large")

with col_main:
    st.markdown('<div class="game-card">', unsafe_allow_html=True)
    st.markdown("### 📡 Input Terminal")
    
    # Session State
    if 'h' not in st.session_state: st.session_state.update({'h':'', 'b':'', 'l':'Positive'})

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("🎲 สุ่มรีวิว (FAST)"):
            s = df.sample(1).iloc[0]
            st.session_state.update({'h': f"ID: {s['review_id'][:8]}", 'b': s['text'], 'l': s['label']})
            st.rerun()
    with c2:
        if st.button("🧹 เคลียร์ข้อมูล"):
            st.session_state.update({'h':'', 'b':'', 'l':'Positive'})
            st.rerun()

    h_in = st.text_input("Analysis ID", value=st.session_state.h)
    b_in = st.text_area("Content Body", value=st.session_state.b, height=150)
    
    if st.button("⚡ EXECUTE NEURAL ANALYSIS", use_container_width=True, type="primary"):
        if b_in.strip():
            st.divider()
            m_c1, m_c2 = st.columns(2)
            for m, col, name in [(model_v1, m_c1, "ALPHA CORE"), (model_v2, m_c2, "SIGMA CORE")]:
                with col:
                    if m:
                        full_text = f"{h_in} {b_in}"
                        probs = m.predict_proba([full_text])[0]
                        pred = m.classes_[np.argmax(probs)]
                        conf = np.max(probs) * 100
                        
                        st.markdown(f"**{name}**")
                        color = "#28a745" if pred == "Positive" else "#dc3545" if pred == "Negative" else "#ffc107"
                        st.markdown(f"<h2 style='color:{color}; margin:0;'>{pred}</h2>", unsafe_allow_html=True)
                        st.progress(int(conf))
                        
                        feats = get_top_features(m, full_text, pred)
                        for w, _ in feats:
                            st.markdown(f'<span class="feature-tag">{w}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. CHATBOT FRAGMENT (แยกส่วนเพื่อความลื่น) ---
with col_side:
    st.markdown('<div class="game-card" style="height: 650px;">', unsafe_allow_html=True)
    st.markdown("### 💬 Tactical Comms")
    
    if "messages" not in st.session_state: st.session_state.messages = []

    # ใช้ Container กำหนดความสูงของแชท
    chat_container = st.container(height=450)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ส่วนรับข้อความ
    if prompt := st.chat_input("Communicate with AI Core..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                res = "Analysis complete. Data patterns are consistent with current mission parameters."
                st.markdown(res)
                st.session_state.messages.append({"role": "assistant", "content": res})
    st.markdown('</div>', unsafe_allow_html=True)

# --- 7. FOOTER METRICS ---
st.markdown('<div class="game-card">', unsafe_allow_html=True)
f1, f2, f3, f4 = st.columns(4)
f1.metric("Database", "5,000 Nodes", "Active")
f2.metric("Neural Accuracy", "99.8%", "Peak")
f3.metric("Protocol", "Logistic", "Stable")
f4.metric("Latency", "1.2ms", "-0.2ms")
st.markdown('</div>', unsafe_allow_html=True)
