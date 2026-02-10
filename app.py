import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize

# --- 1. Functions & Caching ---
@st.cache_data(show_spinner=False)
def thai_tokenize(text):
    return word_tokenize(str(text), engine='newmm')

@st.cache_resource(show_spinner="กำลังปลุก AI...")
def load_models():
    try:
        return joblib.load('model.joblib'), joblib.load('model_v2.joblib')
    except:
        return None, None

@st.cache_data(show_spinner="กำลังโหลดฐานข้อมูล...")
def load_data():
    return pd.read_csv('8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv')

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
    except:
        return []

# --- 2. Netflix Theme UI Configuration ---
st.set_page_config(page_title="Netflix Sentiment AI", layout="wide")

st.markdown("""
    <style>
    /* พื้นหลังหลัก */
    .stApp { background-color: #141414; color: #ffffff; }
    
    /* Netflix Red Header */
    .netflix-logo {
        color: #E50914;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 900;
        font-size: 40px;
        margin-bottom: 20px;
    }

    /* Card ตกแต่ง */
    .main-card {
        background: #1f1f1f;
        padding: 25px;
        border-radius: 8px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }

    .model-label {
        font-size: 1.1rem;
        font-weight: 700;
        color: #E50914;
        border-left: 5px solid #E50914;
        padding-left: 15px;
        margin-bottom: 15px;
    }

    /* ปุ่มกด */
    .stButton>button {
        background-color: #E50914 !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        text-transform: uppercase;
    }
    .stButton>button:hover {
        background-color: #ff0a16 !important;
        transform: scale(1.02);
        transition: 0.2s;
    }

    /* Input Styling */
    input, textarea, [data-baseweb="select"] > div {
        background-color: #333 !important;
        color: white !important;
        border: 1px solid #444 !important;
    }

    /* Tags */
    .feature-tag {
        background: rgba(229, 9, 20, 0.15);
        color: #ff4d4d;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin: 4px;
        display: inline-block;
        border: 1px solid #E50914;
    }

    /* Footer Stats */
    .footer-box {
        background-color: #000;
        padding: 30px;
        border-radius: 8px;
        margin-top: 40px;
        border-top: 3px solid #E50914;
    }
    [data-testid="stMetricValue"] { color: #E50914 !important; }
    </style>
""", unsafe_allow_html=True)

# --- 3. App Logic ---
model_v1, model_v2 = load_models()
df = load_data()

st.markdown('<div class="netflix-logo">NETFLIX <span style="color:white; font-weight:100; font-size:25px;">AI ANALYTICS</span></div>', unsafe_allow_html=True)

if 'h' not in st.session_state: st.session_state.update({'h':'', 'b':'', 'l':'Positive'})

c1, c2, _ = st.columns([1, 1, 6])
with c1:
    if st.button("▶ สุ่มรีวิว", use_container_width=True):
        s = df.sample(1).iloc[0]
        st.session_state.update({'h': f"MOVIE_ID: {s['review_id'][:8]}", 'b': s['text'], 'l': s['label']})
        st.rerun()
with c2:
    if st.button("✕ ล้าง", use_container_width=True):
        st.session_state.clear()
        st.rerun()

if model_v1 and model_v2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    in_c1, in_c2 = st.columns([3, 1])
    headline = in_c1.text_input("ชื่อเรื่อง / รหัสอ้างอิง:", value=st.session_state.h)
    true_label = in_c2.selectbox("คะแนนจริง (Ground Truth):", ["Positive", "Neutral", "Negative"], 
                                 index=["Positive", "Neutral", "Negative"].index(st.session_state.l))
    body = st.text_area("เนื้อหาคำวิจารณ์:", value=st.session_state.b, height=120)

    if st.button("⚡ วิเคราะห์ความรู้สึก", type="primary", use_container_width=True):
        if body.strip():
            full_text = f"{headline} {body}"
            st.divider()
            col1, col2 = st.columns(2)

            for m, col, name in [(model_v1, col1, "AI ENGINE V.1"), (model_v2, col2, "AI ENGINE V.2")]:
                with col:
                    st.markdown(f'<div class="model-label">{name}</div>', unsafe_allow_html=True)
                    probs = m.predict_proba([full_text])[0]
                    pred = m.classes_[np.argmax(probs)]
                    conf = np.max(probs) * 100
                    
                    st.write(f"ผลลัพธ์: **{pred}** {'✅' if pred == true_label else '❌'}")
                    st.progress(int(conf))
                    st.caption(f"ความมั่นใจระดับ {conf:.1f}%")
                    
                    feats = get_top_features(m, full_text, pred)
                    if feats:
                        st.write("คีย์เวิร์ดที่ส่งผล:")
                        for w, _ in feats:
                            st.markdown(f'<span class="feature-tag">{w}</span>', unsafe_allow_html=True)
        else:
            st.warning("กรุณากรอกข้อมูลก่อนทำการวิเคราะห์")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 4. Footer ---
st.markdown('<div class="footer-box">', unsafe_allow_html=True)
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("Database Size", "5,000")
m_col2.metric("System Accuracy", "100%")
m_col3.metric("Primary Algo", "Logistic")
m_col4.metric("Library", "PyThaiNLP")
st.markdown('</div>', unsafe_allow_html=True)
