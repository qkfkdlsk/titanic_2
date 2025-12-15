import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit 앱 제목 설정
st.title("🚢 타이타닉 생존자 분석 (Pclass 및 Age)")
st.markdown("---")

# 데이터 로드 및 전처리 (다중 인코딩 및 Python 엔진 시도)
@st.cache_data
def load_data(file_path):
    """
    CSV 파일을 로드하고 필요한 전처리를 수행합니다.
    인코딩 및 토큰화 오류를 해결하기 위해 다중 인코딩과 Python 엔진을 사용합니다.
    """
    # 콤마(,) 구분자와 파이썬 엔진을 사용하여 파싱 오류를 방지합니다.
    # 인코딩 문제 해결을 위해 여러 인코딩을 순서대로 시도합니다.
    ENCODINGS = ['cp1252', 'latin-1', 'ISO-8859-1', 'utf-8']
    df = None
    
    for encoding in ENCODINGS:
        try:
            # ⭐ 핵심 수정: engine='python'과 sep=',' 명시
            # Python 엔진은 C 엔진보다 복잡한 CSV 구조에 더 강합니다.
            df = pd.read_csv(file_path, encoding=encoding, sep=',', engine='python')
            st.success(f"데이터를 '{encoding}' 인코딩과 Python 엔진으로 성공적으로 로드했습니다.")
            break  # 로드에 성공하면 반복을 중단합니다.
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError as pe:
            # 토큰화 오류가 발생하더라도, 일단 인코딩을 계속 시도합니다.
            # 하지만 Python 엔진 사용 시 이 오류는 발생하지 않을 가능성이 높습니다.
            continue
        except Exception as e:
            # 기타 오류 처리
            st.error(f"데이터 로드 중 예상치 못한 오류가 발생했습니다: {e}")
            return None

    if df is None:
        st.error("💔 로드 실패: 모든 시도(인코딩/파서)에도 불구하고 파일을 읽을 수 없습니다.")
        st.error("해결책: 데이터 파일을 메모장이나 텍스트 편집기로 열어 내용을 확인하거나, **UTF-8 인코딩**으로 변환 후 다시 시도해 주십시오.")
        return None

    # --- 데이터 전처리 시작 ---
    # 컬럼 이름 통일: pclass -> Pclass, survived -> Survived
    df.columns = [col.lower() for col in df.columns]
    df.rename(columns={'pclass': 'Pclass', 'survived': 'Survived'}, inplace=True)
    
    # Age 결측치 처리 (중앙값으로 대체)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Survived와 Pclass 컬럼을 정수형으로 변환
    df['Survived'] = df['Survived'].astype(int)
    df['Pclass'] = df['Pclass'].astype(int)
    
    return df

# 사용자 지정 파일 경로 (이름이 정확한지 다시 한번 확인해 주세요)
FILE_PATH = "titanic.xls - titanic3.csv" 
data = load_data(FILE_PATH)

if data is not None:
    st.header("📋 원본 데이터 미리보기")
    st.dataframe(data.head())
    st.markdown("---")

    # ... (나머지 Pclass 및 Age 분석 코드는 동일)
    
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
