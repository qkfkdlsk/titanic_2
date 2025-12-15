import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🚢 타이타닉 생존자 분석 (Pclass 및 Age)")
st.markdown("---")

# 데이터 로드 및 전처리 (다중 인코딩 및 다중 구분자 시도)
@st.cache_data
def load_data(file_path):
    """
    CSV 파일을 로드하고 필요한 전처리를 수행합니다.
    인코딩 및 토큰화 오류를 해결하기 위해 다중 인코딩, Python 엔진, 다중 구분자를 사용합니다.
    """
    # Excel CSV 파일에서 흔히 발생하는 인코딩과 구분자를 정의합니다.
    ENCODINGS = ['cp1252', 'latin-1', 'utf-8']
    DELIMITERS = [',', ';', '\t']  # 콤마, 세미콜론, 탭
    df = None
    
    # 모든 조합을 시도합니다.
    for encoding in ENCODINGS:
        for delimiter in DELIMITERS:
            try:
                # ⭐ 핵심 수정: engine='python', sep을 현재 구분자로 설정
                # Python 엔진은 복잡한 CSV 구조에 강하며, 다양한 구분자를 시도합니다.
                df = pd.read_csv(file_path, encoding=encoding, sep=delimiter, engine='python')
                
                # 데이터가 최소한의 구조를 갖는지 확인 (컬럼 수가 10개 이상인지 확인)
                if df.shape[1] >= 10:
                    st.success(f"데이터를 '{encoding}' 인코딩과 구분자 '{delimiter}'로 성공적으로 로드했습니다.")
                    break  # 로드에 성공하면 반복을 중단합니다.
                
                # 만약 로드에 성공했으나 컬럼 수가 너무 적다면 (파싱 실패의 징후), 다음 시도로 넘어갑니다.
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            except Exception as e:
                # 기타 오류 처리 (파일 경로 오류 등)
                st.error(f"데이터 로드 중 예상치 못한 오류가 발생했습니다: {e}")
                return None
        if df is not None and df.shape[1] >= 10:
            break
    
    if df is None:
        st.error("💔 로드 실패: 시도한 모든 조합(인코딩/구분자)으로 파일을 읽을 수 없습니다.")
        st.error("해결책: 데이터 파일을 메모장/VS Code로 열어 **UTF-8 인코딩**으로 '다른 이름으로 저장'하거나, 실제 구분자가 콤마나 세미콜론이 아닌지 확인해 주십시오.")
        return None

    # --- 데이터 전처리 시작 ---
    
    # 컬럼 이름 통일: pclass -> Pclass, survived -> Survived
    df.columns = [col.lower() for col in df.columns]
    df.rename(columns={'pclass': 'Pclass', 'survived': 'Survived'}, inplace=True)
    
    # Age 결측치 처리 (중앙값으로 대체)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # 'Survived'와 'Pclass' 컬럼을 정수형으로 변환 (NaN으로 인해 float으로 로드되었을 수 있음)
    if 'Survived' in df.columns and 'Pclass' in df.columns:
        df['Survived'] = df['Survived'].fillna(0).astype(int) # 결측치는 0으로 채우고 정수형으로 변환
        df['Pclass'] = df['Pclass'].fillna(3).astype(int)     # 결측치는 3등석으로 채우고 정수형으로 변환
    
    return df

# 사용자 지정 파일 경로
# 파일 이름을 변경했다면 아래를 수정하십시오! (예: "titanic3.csv")
FILE_PATH = "titanic.csv" 
data = load_data(FILE_PATH)

if data is not None:
    st.header("📋 원본 데이터 미리보기")
    st.dataframe(data.head())
    st.markdown("---")

    ## 1. Pclass별 생존자 비율 분석
    st.header("1️⃣ Pclass (객실 등급)별 생존자 비율")

    # Pclass별 생존자 비율 계산
    pclass_survival = data.groupby('Pclass')['Survived'].agg(['sum', 'count']).reset_index()
    pclass_survival.columns = ['Pclass', 'Survivors', 'Total']
    pclass_survival['Survival Rate (%)'] = (pclass_survival['Survivors'] / pclass_survival['Total']) * 100

    # 결과 테이블 표시 및 시각화
    st.dataframe(pclass_survival.set_index('Pclass').style.format({'Survival Rate (%)': '{:.2f}%'}))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Pclass', y='Survival Rate (%)', data=pclass_survival, palette='viridis', ax=ax)
    ax.set_title('Survival Rate by Passenger Class', fontsize=16)
    ax.set_xlabel('Passenger Class (객실 등급)', fontsize=12)
    ax.set_ylabel('Survival Rate (%) (생존 비율)', fontsize=12)
    st.pyplot(fig)


    ## 2. Age별 생존자 비율 분석
    st.header("2️⃣ Age (나이) 그룹별 생존자 비율")

    # Age 그룹을 위한 Bin 생성
    bins = [0, 12, 18, 35, 60, 100]
    labels = ['Child (0-11)', 'Teen (12-17)', 'Young Adult (18-34)', 'Adult (35-59)', 'Senior (60+)']
    data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False, include_lowest=True)

    # AgeGroup별 생존자 비율 계산
    age_survival = data.groupby('AgeGroup', observed=True)['Survived'].agg(['sum', 'count']).reset_index()
    age_survival.columns = ['AgeGroup', 'Survivors', 'Total']
    age_survival['Survival Rate (%)'] = (age_survival['Survivors'] / age_survival['Total']) * 100
    
    # 결과 테이블 표시 및 시각화
    st.dataframe(age_survival.set_index('AgeGroup').style.format({'Survival Rate (%)': '{:.2f}%'}))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='AgeGroup', y='Survival Rate (%)', data=age_survival, palette='plasma', ax=ax)
    ax.set_title('Survival Rate by Age Group', fontsize=16)
    ax.set_xlabel('Age Group (나이 그룹)', fontsize=12)
    ax.set_ylabel('Survival Rate (%) (생존 비율)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
