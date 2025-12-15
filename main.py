import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🚢 타이타닉 생존자 분석 (Pclass 및 Age)")
st.markdown("---")

# 데이터 로드 및 전처리 (최종 안정화 버전)
@st.cache_data
def load_data(file_path):
    """
    CSV 파일을 로드하고 필요한 전처리를 수행합니다.
    인코딩 및 파싱 오류를 해결하기 위해 다중 인코딩/구분자를 시도하고,
    KeyError 방지를 위해 컬럼 이름을 정리합니다.
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
                
                # 로드 성공 후, 컬럼 개수 확인 (Titanic 데이터는 약 14개 컬럼)
                if df.shape[1] >= 10:
                    st.success(f"데이터를 '{encoding}' 인코딩과 구분자 '{delimiter}'로 성공적으로 로드했습니다.")
                    break 
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            except Exception as e:
                # 파일 경로 오류 등
                # st.error(f"데이터 로드 중 예상치 못한 오류 발생: {e}")
                return None
        if df is not None and df.shape[1] >= 10:
            break
    
    if df is None:
        st.error("💔 로드 실패: 모든 시도에도 불구하고 파일을 읽을 수 없습니다. 파일의 인코딩/구분자를 수동으로 확인해 주십시오.")
        return None

    # --- 데이터 전처리 시작 (KeyError 방지) ---
    
    # ⭐ 핵심 수정 1: 컬럼 이름의 공백 제거 및 소문자화
    # DataFrame의 모든 컬럼 이름을 소문자로 만들고, 앞뒤 공백을 제거합니다.
    df.columns = [col.strip().lower() for col in df.columns]
    
    # ⭐ 핵심 수정 2: 분석에 사용할 컬럼 이름 명확히 정의
    required_cols = {'pclass': 'Pclass', 'survived': 'Survived', 'age': 'Age'}
    rename_map = {}
    
    for lower_name, capitalized_name in required_cols.items():
        if lower_name in df.columns:
            rename_map[lower_name] = capitalized_name
        else:
            st.error(f"Error: 필수 컬럼 '{lower_name}' (객실 등급, 생존 여부, 나이 중 하나)가 데이터에 없습니다.")
            return None
            
    df.rename(columns=rename_map, inplace=True)
    
    # Age 결측치 처리 (중앙값으로 대체)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Survived와 Pclass 컬럼을 정수형으로 변환
    df['Survived'] = df['Survived'].fillna(0).astype(int)
    df['Pclass'] = df['Pclass'].fillna(3).astype(int)
    
    return df

# 사용자 지정 파일 경로
# 🚨 파일 이름을 'titanic3.csv'로 변경했다면, 아래를 수정해야 합니다.
FILE_PATH = "titanic.xls - titanic3.csv" 
# 혹은 안전하게: FILE_PATH = "titanic3.csv"

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
