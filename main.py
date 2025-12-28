import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from matplotlib import font_manager

# --- 1. 폰트 객체 생성 함수 ---
@st.cache_resource # 폰트 로드는 한 번만 수행하도록 캐싱
def get_font():
    try:
        f_list = font_manager.findSystemFonts()
        # Linux(Streamlit Cloud) 환경에서 나눔고딕 찾기
        font_path = next((f for f in f_list if 'nanumgothic' in f.lower().replace(" ", "")), None)
        
        # 못 찾을 경우 나눔 계열 아무거나 찾기
        if not font_path:
            font_path = next((f for f in f_list if 'nanum' in f.lower()), None)
            
        if font_path:
            return font_manager.FontProperties(fname=font_path)
    except:
        pass
    return None

# --- 2. 데이터 로드 (이전과 동일) ---
@st.cache_data
def load_data(file_path):
    try:
        # 다양한 인코딩 시도 (가장 깔끔한 로직)
        df = pd.read_csv(file_path, encoding='cp1252') 
        df.columns = df.columns.str.replace('ï»¿', '', regex=False).str.strip().lower()
        # 컬럼명 표준화
        rename_dict = {'pclass': 'Pclass', 'survived': 'Survived', 'age': 'Age'}
        df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns}, inplace=True)
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Survived'] = df['Survived'].fillna(0).astype(int)
        return df
    except:
        return None

# --- 메인 실행부 ---
st.title("🚢 타이타닉 생존자 분석")
st.markdown("---")

data = load_data("titanic3.csv")
font_prop = get_font()

if data is not None:
    if font_prop:
        st.success(f"✅ 폰트 로드 완료: {font_prop.get_name()}")
    else:
        st.error("❌ 폰트를 찾지 못했습니다. 'packages.txt'를 확인해 주세요.")

    # --- 1️⃣ 객실 등급별 생존율 ---
    st.header("1️⃣ 객실 등급(Pclass)별 생존율")
    pclass_survival = data.groupby('Pclass')['Survived'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Pclass', y='Survived', data=pclass_survival, palette='viridis', ax=ax)
    
    # ⭐ 폰트 직접 주입 (이 부분이 핵심)
    if font_prop:
        ax.set_title('객실 등급별 생존율', fontproperties=font_prop, fontsize=18)
        ax.set_xlabel('객실 등급 (1, 2, 3등석)', fontproperties=font_prop, fontsize=12)
        ax.set_ylabel('생존율 (0.0 ~ 1.0)', fontproperties=font_prop, fontsize=12)
    st.pyplot(fig)

    # --- 2️⃣ 나이 그룹별 생존율 ---
    st.header("2️⃣ 나이 그룹별 생존율")
    bins = [0, 12, 18, 35, 60, 100]
    labels = ['어린이', '청소년', '청년', '성인', '노년']
    data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
    age_survival = data.groupby('AgeGroup', observed=True)['Survived'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='AgeGroup', y='Survived', data=age_survival, palette='plasma', ax=ax)
    
    # ⭐ 폰트 직접 주입
    if font_prop:
        ax.set_title('나이 그룹별 생존율', fontproperties=font_prop, fontsize=18)
        ax.set_xlabel('나이 그룹', fontproperties=font_prop, fontsize=12)
        ax.set_ylabel('생존율', fontproperties=font_prop, fontsize=12)
        # X축 눈금(어린이, 청소년 등) 한글 처리
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_prop)
    st.pyplot(fig)
