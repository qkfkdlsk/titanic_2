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
    다중 인코딩/구분자를 시도하고, KeyError 방지를 위해 컬럼 이름 정리 후 확인합니다.
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
                    st.success(f"데이터 로드 성공: '{encoding}' 인코딩과 구분자 '{delimiter}' 사용")
                    break 
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            except Exception:
                continue
        if df is not None and df.shape[1] >= 10:
            break
    
    if df is None:
        st.error("💔 로드 실패: 모든 시도에도 불구하고 파일을 읽을 수 없습니다.")
        return None

    # --- 데이터 전처리 시작 (KeyError 방지) ---
    
    # 컬럼 이름의 공백 제거 및 소문자화 (KeyError 방지 1)
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

    # ⭐ 핵심 디버그: 필수 컬럼이 누락된 경우, 실제 컬럼 목록을 출력
    if missing_cols:
        st.error(f"⚠️ **분석 실패:** 필수 컬럼이 데이터에 없습니다.")
        st.error(f"누락된 필수 컬럼(소문자 기준): {', '.join(missing_cols)}")
        st.write("---")
        st.subheader("🧐 데이터 파일에 실제 존재하는 컬럼 목록:")
        st.dataframe(pd.DataFrame({'Actual Columns': df.columns.tolist()}))
        return None # 필수 컬럼이 없으므로 분석 중단
    
    # 컬럼 이름 변경 및 나머지 전처리
    df.rename(columns=rename_map, inplace=True)
    
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Survived'] = df['Survived'].fillna(0).astype(int)
    df['Pclass'] = df['Pclass'].fillna(3).astype(int)
    
    return df

# 사용자 지정 파일 경로
# 파일 이름을 변경했다면 아래를 수정하십시오! (예: "titanic3.csv")
FILE_PATH = "titanic.xls - titanic3.csv" 
data = load_data(FILE_PATH)

if data is not None:
    # 이 아래 블록이 실행되면 분석 결과가 정상적으로 출력됩니다.
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
