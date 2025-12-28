import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import os

# --- 1. 폰트 설정 함수 (더 강력한 방식) ---
def get_korean_font():
    try:
        from matplotlib import font_manager, rc
        
        f_list = font_manager.findSystemFonts()
        font_path = None
        
        # OS별 폰트 파일 찾기
        if platform.system() == 'Windows':
            font_path = next((f for f in f_list if 'malgun' in f.lower()), None)
        elif platform.system() == 'Darwin': # macOS
            return font_manager.FontProperties(family='AppleGothic')
        else: # Linux (Streamlit Cloud)
            # 나눔고딕을 우선 찾고 없으면 나눔바른고딕 등을 찾음
            font_path = next((f for f in f_list if 'nanumgothic' in f.lower().replace(" ", "")), None)
            if not font_path:
                font_path = next((f for f in f_list if 'nanum' in f.lower()), None)
        
        if font_path:
            # 폰트 프로퍼티 객체 반환
            return font_manager.FontProperties(fname=font_path)
        return None
    except Exception as e:
        st.error(f"폰트 로드 중 오류: {e}")
        return None

# --- 2. 페이지 설정 ---
st.title("🚢 타이타닉 생존자 분석")
st.markdown("---")

FILE_PATH = "titanic3.csv" 

@st.cache_data
def load_data(file_path):
    # (이전과 동일한 로드 로직...)
    try:
        df = pd.read_csv(file_path, encoding='cp1252') # 혹은 자동 인코딩 로직 사용
        df.columns = df.columns.str.replace('ï»¿', '', regex=False).str.strip().lower()
        df.rename(columns={'pclass': 'Pclass', 'survived': 'Survived', 'age': 'Age'}, inplace=True)
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Survived'] = df['Survived'].fillna(0).astype(int)
        return df
    except:
        return None

data = load_data(FILE_PATH)

if data is not None:
    # 폰트 객체 가져오기
    font_prop = get_korean_font()
    if font_prop:
        st.info(f"✅ 사용 중인 폰트: {font_prop.get_name()}")
    else:
        st.warning("⚠️ 한글 폰트를 찾지 못했습니다.")

    # 1️⃣ Pclass별 생존율
    st.header("1️⃣ 객실 등급(Pclass)별 생존율")
    pclass_survival = data.groupby('Pclass')['Survived'].mean() * 100
    pclass_survival = pclass_survival.reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Pclass', y='Survived', data=pclass_survival, palette='viridis', ax=ax)
    
    # ⭐ 폰트 직접 적용 (이 부분이 핵심입니다)
    if font_prop:
        ax.set_title('객실 등급별 생존율 (%)', fontproperties=font_prop, fontsize=16)
        ax.set_xlabel('객실 등급 (1=1등석, 2=2등석, 3=3등석)', fontproperties=font_prop, fontsize=12)
        ax.set_ylabel('생존율 (%)', fontproperties=font_prop, fontsize=12)
    
    st.pyplot(fig)

    # 2️⃣ 나이 그룹별 생존율
    st.header("2️⃣ 나이 그룹(Age Group)별 생존율")
    bins = [0, 12, 18, 35, 60, 100]
    labels = ['어린이', '청소년', '청년', '성인', '노년']
    data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

    age_survival = data.groupby('AgeGroup', observed=True)['Survived'].mean() * 100
    age_survival = age_survival.reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='AgeGroup', y='Survived', data=age_survival, palette='plasma', ax=ax)
    
    # ⭐ 폰트 직접 적용
    if font_prop:
        ax.set_title('나이 그룹별 생존율 (%)', fontproperties=font_prop, fontsize=16)
        ax.set_xlabel('나이 그룹', fontproperties=font_prop, fontsize=12)
        ax.set_ylabel('생존율 (%)', fontproperties=font_prop, fontsize=12)
        # X축 눈금 한글 처리
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_prop)
    
    st.pyplot(fig)
