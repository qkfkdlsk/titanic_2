네, 요청하신 대로 시스템에 설치된 폰트를 자동으로 찾아 적용하는 로직을 합쳐서 app.py 전체 코드를 다시 정리해 드립니다.

이 코드는 특정 경로('/usr/share/...')를 강제하지 않고, 시스템(Streamlit Cloud 리눅스 서버 등) 내에서 나눔고딕을 검색하여 연결하기 때문에 훨씬 안정적입니다.

🚀 최종 완성된 app.py 코드
Python

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# --- 1. 폰트 설정 및 라이브러리 임포트 (데이터 로드 후에 실행되도록 블록화) ---
def setup_korean_font():
    try:
        from matplotlib import font_manager, rc
        import matplotlib as mpl
        
        # 시스템에 설치된 전체 폰트 리스트 가져오기
        f_list = font_manager.findSystemFonts()
        font_path = None
        
        # OS별 폰트 찾기 로직
        if platform.system() == 'Windows':
            font_path = next((f for f in f_list if 'malgun' in f.lower()), None)
        elif platform.system() == 'Darwin': # macOS
            rc('font', family='AppleGothic')
            return True
        else: # Linux (Streamlit Cloud)
            # 'nanumgothic' 단어가 포함된 폰트 파일 찾기
            font_path = next((f for f in f_list if 'nanumgothic' in f.lower().replace(" ", "")), None)
        
        if font_path:
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            rc('font', family=font_name)
            plt.rcParams['axes.unicode_minus'] = False 
            st.info(f"✅ 그래프 폰트 설정을 완료했습니다: {font_name}")
            return True
        else:
            st.warning("⚠️ 시스템에서 한글 폰트(나눔고딕/맑은고딕)를 찾을 수 없습니다. 'packages.txt'를 확인해 주세요.")
            return False
            
    except Exception as e:
        st.error(f"❌ 폰트 설정 중 오류 발생: {e}")
        return False

# --- 2. 페이지 타이틀 ---
st.title("🚢 타이타닉 생존자 분석 (Pclass 및 Age)")
st.markdown("---")

# 🚨 파일 이름 설정
FILE_PATH = "titanic3.csv" 

# --- 3. 데이터 로드 및 전처리 함수 ---
@st.cache_data
def load_data(file_path):
    ENCODINGS = ['cp1252', 'latin-1', 'utf-8']
    DELIMITERS = [',', ';', '\t']
    df = None
    
    for encoding in ENCODINGS:
        for delimiter in DELIMITERS:
            try:
                df = pd.read_csv(file_path, encoding=encoding, sep=delimiter, engine='python')
                if df is not None and df.shape[1] >= 10:
                    st.success(f"✅ 데이터 로드 성공: '{encoding}' 인코딩 사용")
                    break 
            except:
                continue
        if df is not None: break
    
    if df is None:
        st.error("💔 파일을 읽을 수 없습니다. 파일명과 경로를 확인해 주세요.")
        return None

    # BOM 제거 및 컬럼명 정리
    df.columns = df.columns.str.replace('ï»¿', '', regex=False)
    df.columns = [col.strip().lower() for col in df.columns]
    
    # 필수 컬럼 확인 및 이름 변경
    required_cols = {'pclass': 'Pclass', 'survived': 'Survived', 'age': 'Age'}
    rename_map = {}
    for lower_name, capitalized_name in required_cols.items():
        if lower_name in df.columns:
            rename_map[lower_name] = capitalized_name
        else:
            st.error(f"⚠️ 필수 컬럼 '{lower_name}'이 없습니다.")
            return None
    
    df.rename(columns=rename_map, inplace=True)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Survived'] = df['Survived'].fillna(0).astype(int)
    df['Pclass'] = df['Pclass'].fillna(3).astype(int)
    
    return df

# --- 4. 메인 실행 블록 ---
data = load_data(FILE_PATH)

if data is not None:
    # 데이터 로드 후 폰트 설정 실행
    setup_korean_font()

    st.header("📋 원본 데이터 미리보기")
    st.dataframe(data.head())
    st.markdown("---")

    # 1️⃣ Pclass별 생존자 비율 분석
    st.header("1️⃣ 객실 등급(Pclass)별 생존율")
    pclass_survival = data.groupby('Pclass')['Survived'].mean() * 100
    pclass_survival = pclass_survival.reset_index()
    pclass_survival.columns = ['객실 등급', '생존율 (%)']

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='객실 등급', y='생존율 (%)', data=pclass_survival, palette='viridis', ax=ax)
    ax.set_title('객실 등급별 생존율 (%)', fontsize=15)
    st.pyplot(fig)

    # 2️⃣ 나이 그룹별 생존자 비율 분석
    st.header("2️⃣ 나이 그룹(Age Group)별 생존율")
    bins = [0, 12, 18, 35, 60, 100]
    labels = ['어린이', '청소년', '청년', '성인', '노년']
    data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

    age_survival = data.groupby('AgeGroup', observed=True)['Survived'].mean() * 100
    age_survival = age_survival.reset_index()
    age_survival.columns = ['나이 그룹', '생존율 (%)']

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='나이 그룹', y='생존율 (%)', data=age_survival, palette='plasma', ax=ax)
    ax.set_title('나이 그룹별 생존율 (%)', fontsize=15)
    st.pyplot(fig)
