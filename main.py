import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform
# matplotlib의 font_manager와 rc는 아래 폰트 설정 블록에서 임포트합니다.

st.title("🚢 타이타닉 생존자 분석 (Pclass 및 Age)")
st.markdown("---")

# 🚨 파일 이름 설정 (파일명이 'titanic3.csv'인지 다시 한번 확인해 주세요)
FILE_PATH = "titanic3.csv" 

# 데이터 로드 및 전처리 함수
@st.cache_data
def load_data(file_path):
    ENCODINGS = ['cp1252', 'latin-1', 'utf-8']
    DELIMITERS = [',', ';', '\t']
    df = None
    
    for encoding in ENCODINGS:
        for delimiter in DELIMITERS:
            try:
                df = pd.read_csv(file_path, encoding=encoding, sep=delimiter, engine='python')
                if df.shape[1] >= 10 and not df.empty:
                    st.success(f"✅ 데이터 로드 성공: '{encoding}' 인코딩, 구분자 '{delimiter}' 사용")
                    break 
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            except FileNotFoundError:
                st.error(f"❌ 파일 경로/이름 오류: '{file_path}' 파일을 찾을 수 없습니다.")
                return None
            except Exception as e:
                st.warning(f"경고: 로드 중 오류 발생 ({encoding}, {delimiter}): {e}")
                continue
        if df is not None and df.shape[1] >= 10 and not df.empty:
            break
    
    if df is None or df.empty:
        st.error("💔 로드 실패: 모든 시도에도 불구하고 파일을 읽을 수 없습니다.")
        return None

    # --- 데이터 전처리 시작 (BOM 및 KeyError 방지) ---
    df.columns = df.columns.str.replace('ï»¿', '', regex=False)
    df.columns = [col.strip().lower() for col in df.columns]
    
    required_cols = {'pclass': 'Pclass', 'survived': 'Survived', 'age': 'Age'}
    rename_map = {}
    missing_cols = []
    
    for lower_name, capitalized_name in required_cols.items():
        if lower_name in df.columns:
            rename_map[lower_name] = capitalized_name
        else:
            missing_cols.append(lower_name)

    if missing_cols:
        st.error(f"⚠️ **분석 실패:** 필수 컬럼이 데이터에 없습니다. 누락: {', '.join(missing_cols)}")
        st.subheader("🧐 현재 데이터에 실제 존재하는 컬럼 목록:")
        st.dataframe(pd.DataFrame({'Actual Columns': df.columns.tolist()}))
        return None 
    
    df.rename(columns=rename_map, inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Survived'] = df['Survived'].fillna(0).astype(int)
    df['Pclass'] = df['Pclass'].fillna(3).astype(int)
    
    return df

data = load_data(FILE_PATH)

if data is not None:
    
    # ⭐⭐⭐ 폰트 설정 코드: Linux 환경의 NanumGothic을 강제 참조합니다. ⭐⭐⭐
    try:
        from matplotlib import font_manager, rc # 여기서 임포트
        
        # 1. 폰트 경로를 Linux 표준 경로로 직접 지정 (packages.txt 사용 시)
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf' 
        
        # 2. 시스템 환경별 대체 설정
        if platform.system() == 'Windows':
            font_path = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
            font_path = next((f for f in font_path if 'malgun' in f.lower()), None)
            if font_path is None:
                raise FileNotFoundError("Malgun Gothic 폰트 파일을 찾을 수 없습니다.")
        elif platform.system() == 'Darwin': # macOS
            rc('font', family='AppleGothic')
        
        if font_path and platform.system() != 'Darwin':
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            rc('font', family=font_name)
        
        plt.rcParams['axes.unicode_minus'] = False 
        st.info(f"✅ 그래프 폰트 설정을 완료했습니다. (OS: {platform.system()})")
        
    except Exception as e:
        # 폰트 오류가 발생해도 분석은 계속 진행됩니다.
        st.error(f"❌ 폰트 설정 중 오류가 발생했습니다. 한글이 깨질 수 있습니다. 오류: {e}")

    # --- 분석 및 시각화 시작 ---
        
    st.header("📋 원본 데이터 미리보기")
    st.dataframe(data.head())
    st.markdown("---")

    ## 1. Pclass별 생존자 비율 분석
    st.header("1️⃣ Pclass (객실 등급)별 생존자 비율")

    pclass_survival = data.groupby('Pclass')['Survived'].agg(['sum', 'count']).reset_index()
    pclass_survival.columns = ['Pclass', 'Survivors', 'Total']
    pclass_survival['Survival Rate (%)'] = (pclass_survival['Survivors'] / pclass_survival['Total']) * 100

    st.dataframe(pclass_survival.set_index('Pclass').style.format({'Survival Rate (%)': '{:.2f}%'}))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Pclass', y='Survival Rate (%)', data=pclass_survival, palette='viridis', ax=ax)
    ax.set_title('Passenger Class별 Survival Rate', fontsize=16)
    ax.set_xlabel('객실 등급', fontsize=12)
    ax.set_ylabel('생존 비율 (%)', fontsize=12)
    st.pyplot(fig)


    ## 2. Age별 생존자 비율 분석
    st.header("2️⃣ Age (나이) 그룹별 생존자 비율")

    bins = [0, 12, 18, 35, 60, 100]
    labels = ['어린이 (0-11)', '청소년 (12-17)', '청년 (18-34)', '성인 (35-59)', '노년 (60+)']
    data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False, include_lowest=True)

    age_survival = data.groupby('AgeGroup', observed=True)['Survived'].agg(['sum', 'count']).reset_index()
    age_survival.columns = ['AgeGroup', 'Survivors', 'Total']
    age_survival['Survival Rate (%)'] = (age_survival['Survivors'] / age_survival['Total']) * 100
    
    st.dataframe(age_survival.set_index('AgeGroup').style.format({'Survival Rate (%)': '{:.2f}%'}))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='AgeGroup', y='Survival Rate (%)', data=age_survival, palette='plasma', ax=ax)
    ax.set_title('나이 그룹별 Survival Rate', fontsize=16)
    ax.set_xlabel('나이 그룹', fontsize=12)
    ax.set_ylabel('생존 비율 (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
