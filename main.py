import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit 앱 제목 설정
st.title("🚢 타이타닉 생존자 분석 (Pclass 및 Age)")
st.markdown("---")

# 데이터 로드 및 전처리 (인코딩 수정)
@st.cache_data
def load_data(file_path):
    """
    CSV 파일을 로드하고 필요한 전처리를 수행합니다.
    Excel에서 저장된 파일 오류를 해결하기 위해 'cp1252' 인코딩을 우선 시도합니다.
    """
    try:
        # 1. 가장 흔한 Excel 저장 인코딩인 'cp1252'를 먼저 시도 (서유럽/미국 윈도우 기본값)
        df = pd.read_csv(file_path, encoding='cp1252')
        
    except UnicodeDecodeError:
        # 2. cp1252로도 실패하면, 가장 관대한 'latin-1'을 시도합니다.
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
        except Exception as e:
            # 최종적으로 실패하면 오류 메시지 출력 후 종료
            st.error(f"데이터 로드 중 심각한 오류가 발생했습니다: {e}")
            st.error("두 가지 흔한 인코딩(cp1252, latin-1)을 모두 시도했으나 실패했습니다. 파일이 손상되었거나 매우 특수한 인코딩일 수 있습니다.")
            return None
    
    except Exception as e:
        # 파일 경로/이름 오류 등 다른 오류 처리
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
# 이전 문제 해결을 위해 이 파일 이름이 실제 파일 이름과 정확히 일치해야 합니다.
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
