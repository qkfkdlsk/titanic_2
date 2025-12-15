
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit 앱 제목 설정
st.title("🚢 타이타닉 생존자 분석 (Pclass 및 Age)")
st.markdown("---")

# 데이터 로드 및 전처리
@st.cache_data
def load_data(FILE_PATH):
    """
    CSV 파일을 로드하고 필요한 전처리를 수행합니다.
    """
    try:
        # 파일 로드 (titanic3.csv는 인덱스 컬럼 없이 로드)
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        st.error(f"오류: 파일 '{FILE_PATH}'를 찾을 수 없습니다. 파일 이름을 확인하십시오.")
        return None
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        return None

    # 컬럼 이름 통일: pclass -> Pclass, survived -> Survived
    df.columns = [col.lower() for col in df.columns]
    df.rename(columns={'pclass': 'Pclass', 'survived': 'Survived'}, inplace=True)
    
    # Age 결측치 처리 (중앙값으로 대체)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Survived와 Pclass 컬럼을 정수형으로 변환
    df['Survived'] = df['Survived'].astype(int)
    df['Pclass'] = df['Pclass'].astype(int)
    
    return df

# 사용자 지정 파일 경로
FILE_PATH = "titanic.xls"
data = load_data(FILE_PATH)

if data is not None:
    st.header("📋 원본 데이터 미리보기")
    st.dataframe(data.head())
    st.markdown("---")

    ## 1. Pclass별 생존자 비율 분석
    st.header("1️⃣ Pclass (객실 등급)별 생존자 비율")
    st.markdown("객실 등급(1, 2, 3등석)이 생존율에 미치는 영향을 비교합니다.")

    # Pclass별 생존자 비율 계산
    pclass_survival = data.groupby('Pclass')['Survived'].agg(['sum', 'count']).reset_index()
    pclass_survival.columns = ['Pclass', 'Survivors', 'Total']
    pclass_survival['Survival Rate (%)'] = (pclass_survival['Survivors'] / pclass_survival['Total']) * 100

    # 결과 테이블 표시
    st.dataframe(pclass_survival.set_index('Pclass').style.format({'Survival Rate (%)': '{:.2f}%'}))
    
    # 시각화 (막대 그래프)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Pclass', y='Survival Rate (%)', data=pclass_survival, palette='viridis', ax=ax)
    ax.set_title('Survival Rate by Passenger Class', fontsize=16)
    ax.set_xlabel('Passenger Class (객실 등급)', fontsize=12)
    ax.set_ylabel('Survival Rate (%) (생존 비율)', fontsize=12)
    st.pyplot(fig)
    
    st.markdown("---")


    ## 2. Age별 생존자 비율 분석
    st.header("2️⃣ Age (나이) 그룹별 생존자 비율")
    st.markdown("나이를 그룹으로 나누어 각 그룹의 생존율을 비교합니다.")

    # Age 그룹을 위한 Bin 생성
    bins = [0, 12, 18, 35, 60, 100]
    labels = ['Child (0-11)', 'Teen (12-17)', 'Young Adult (18-34)', 'Adult (35-59)', 'Senior (60+)']
    # 'AgeGroup' 컬럼 추가
    data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False, include_lowest=True)

    # AgeGroup별 생존자 비율 계산
    age_survival = data.groupby('AgeGroup', observed=True)['Survived'].agg(['sum', 'count']).reset_index()
    age_survival.columns = ['AgeGroup', 'Survivors', 'Total']
    age_survival['Survival Rate (%)'] = (age_survival['Survivors'] / age_survival['Total']) * 100
    
    # 결과 테이블 표시
    st.dataframe(age_survival.set_index('AgeGroup').style.format({'Survival Rate (%)': '{:.2f}%'}))
    
    # 시각화 (막대 그래프)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='AgeGroup', y='Survival Rate (%)', data=age_survival, palette='plasma', ax=ax)
    ax.set_title('Survival Rate by Age Group', fontsize=16)
    ax.set_xlabel('Age Group (나이 그룹)', fontsize=12)
    ax.set_ylabel('Survival Rate (%) (생존 비율)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

