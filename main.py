import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ⭐⭐⭐ 한글 폰트 설정 (Windows, macOS, Linux 환경별 대응) ⭐⭐⭐
import platform
from matplotlib import font_manager, rc

# 시스템 환경에 따라 폰트 경로 설정
if platform.system() == 'Darwin': # macOS
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows': # Windows
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
elif platform.system() == 'Linux': # Linux (Colab, 서버 등)
    # 나눔 폰트가 설치되어 있지 않다면 설치가 필요합니다.
    # Colab에서 실행 시: !sudo apt-get install -y fonts-nanum
    # 일반 Linux 서버: sudo apt-get install -y fonts-nanum-extra
    rc('font', family='NanumGothic')

# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False 
# ⭐⭐⭐ 여기까지 폰트 설정 코드 추가 ⭐⭐⭐

st.title("🚢 타이타닉 생존자 분석 (Pclass 및 Age)")
st.markdown("---")

# 사용자 지정 파일 경로를 가장 안전한 이름으로 설정합니다.
# 🚨 파일 이름을 'titanic3.csv'로 변경했다면, 이 이름이 정확해야 합니다.
FILE_PATH = "titanic3.csv" 

# 데이터 로드 및 전처리 (최종 진단 버전)
@st.cache_data
def load_data(file_path):
    """
    CSV 파일을 로드하고 필요한 전처리를 수행합니다.
    인코딩 및 파싱 오류를 해결하기 위해 다중 인코딩/구분자를 시도하고,
    BOM 제거 및 KeyError 방지를 위한 컬럼 정리 로직을 포함합니다.
    """
    ENCODINGS = ['cp1252', 'latin-1', 'utf-8']
    DELIMITERS = [',', ';', '\t']
    df = None
    
    # 모든 조합을 시도하여 파일 로드
    for encoding in ENCODINGS:
        for delimiter in DELIMITERS:
            try:
                # Python 엔진 사용 및 구분자/인코딩 시도
                df = pd.read_csv(file_path, encoding=encoding, sep=delimiter, engine='python')
                
                if df.shape[1] >= 10 and not df.empty:
                    st.success(f"✅ 데이터 로드 성공: '{encoding}' 인코딩, 구분자 '{delimiter}' 사용")
                    st.write(f"DataFrame 크기: {df.shape[0]}행, {df.shape[1]}열")
                    break 
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            except FileNotFoundError:
                # 파일 경로/이름 오류가 발생하면, 다른 시도는 할 필요 없으므로 바로 반환
                st.error(f"❌ 파일 경로/이름 오류: '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인하십시오.")
                return None
            except Exception as e:
                st.warning(f"경고: 로드 중 오류 발생 ({encoding}, {delimiter}): {e}")
                continue
        if df is not None and df.shape[1] >= 10 and not df.empty:
            break
    
    if df is None or df.empty:
        st.error("💔 로드 실패: 모든 인코딩/구분자 시도에도 불구하고 파일을 읽을 수 없거나 데이터가 비어있습니다.")
        return None

    # ⬇️⬇️⬇️ 이 부분이 사용자님께서 찾으시던 전처리 로직입니다. ⬇️⬇️⬇️
    # --- 데이터 전처리 시작 (BOM 및 KeyError 방지) ---
    
    # ⭐ 핵심 수정 1: BOM 문자열 제거 (컬럼명 'ï»¿pclass' 문제 해결)
    df.columns = df.columns.str.replace('ï»¿', '', regex=False)
    
    # 기존 로직: 컬럼 이름의 공백 제거 및 소문자화 
    df.columns = [col.strip().lower() for col in df.columns]
    
    # 분석에 사용할 필수 컬럼 정의
    required_cols = {'pclass': 'Pclass', 'survived': 'Survived', 'age': 'Age'}
    rename_map = {}
    missing_cols = []
    
    # 필수 컬럼이 모두 존재하는지 확인
    for lower_name, capitalized_name in required_cols.items():
        if lower_name in df.columns:
            rename_map[lower_name] = capitalized_name
        else:
            missing_cols.append(lower_name)

    # 최종 진단: 필수 컬럼이 누락된 경우, 실제 컬럼 목록을 출력
    if missing_cols:
        st.error(f"⚠️ **분석 실패:** 필수 컬럼이 데이터에 없습니다.")
        st.error(f"누락된 필수 컬럼(소문자 기준): {', '.join(missing_cols)}")
        st.write("---")
        st.subheader("🧐 현재 데이터에 실제 존재하는 컬럼 목록:")
        st.dataframe(pd.DataFrame({'Actual Columns': df.columns.tolist()}))
        return None 
    
    # 전처리 계속
    df.rename(columns=rename_map, inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Survived'] = df['Survived'].fillna(0).astype(int)
    df['Pclass'] = df['Pclass'].fillna(3).astype(int)
    
    return df
    # ⬆️⬆️⬆️ load_data 함수의 끝입니다. ⬆️⬆️⬆️

data = load_data(FILE_PATH)

if data is not None:
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
    ax.set_title('Survival Rate by Passenger Class', fontsize=16)
    ax.set_xlabel('Passenger Class (객실 등급)', fontsize=12)
    ax.set_ylabel('Survival Rate (%) (생존 비율)', fontsize=12)
    st.pyplot(fig)


    ## 2. Age별 생존자 비율 분석
    st.header("2️⃣ Age (나이) 그룹별 생존자 비율")

    bins = [0, 12, 18, 35, 60, 100]
    labels = ['Child (0-11)', 'Teen (12-17)', 'Young Adult (18-34)', 'Adult (35-59)', 'Senior (60+)']
    data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False, include_lowest=True)

    age_survival = data.groupby('AgeGroup', observed=True)['Survived'].agg(['sum', 'count']).reset_index()
    age_survival.columns = ['AgeGroup', 'Survivors', 'Total']
    age_survival['Survival Rate (%)'] = (age_survival['Survivors'] / age_survival['Total']) * 100
    
    st.dataframe(age_survival.set_index('AgeGroup').style.format({'Survival Rate (%)': '{:.2f}%'}))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='AgeGroup', y='Survival Rate (%)', data=age_survival, palette='plasma', ax=ax)
    ax.set_title('Survival Rate by Age Group', fontsize=16)
    ax.set_xlabel('Age Group (나이 그룹)', fontsize=12)
    ax.set_ylabel('Survival Rate (%) (생존 비율)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
