# 머신러닝 워크플로우 (Machine Learning Workflow)

머신러닝 프로젝트는 단순히 알고리즘을 선택하고 코드 몇 줄을 작성하는 것 이상의 체계적이고 종합적인 과정입니다. 성공적인 머신러닝 프로젝트는 비즈니스 문제의 명확한 이해부터 실제 운영 환경에서의 지속적인 모니터링까지 이어지는 **생명주기(Lifecycle)**를 가집니다.

이 전체 과정을 **머신러닝 워크플로우(ML Workflow)** 또는 **머신러닝 파이프라인(ML Pipeline)**이라고 하며, 이는 데이터 과학 프로젝트의 체계적인 방법론을 제공합니다. 각 단계는 서로 유기적으로 연결되어 있으며, 실제 프로젝트에서는 반복적이고 순환적인 특성을 보입니다.

---

## 1. 문제 정의 (Problem Definition)

### 1.1. 비즈니스 문제의 이해

머신러닝 프로젝트의 성공은 첫 번째 단계인 **문제 정의**에서 시작됩니다. 이 단계에서는 다음과 같은 핵심 질문들에 답해야 합니다:

**핵심 질문들:**
- **무엇을 해결하려고 하는가?** 구체적인 비즈니스 문제나 목표 정의
- **현재 어떤 방식으로 해결하고 있는가?** 기존 해결 방법의 한계점 파악
- **머신러닝이 정말 필요한가?** 문제의 복잡성과 데이터 가용성 평가
- **성공을 어떻게 측정할 것인가?** 명확한 평가 기준과 KPI 설정

### 1.2. 머신러닝 문제로 변환

비즈니스 문제를 구체적인 머신러닝 문제로 변환해야 합니다:

**문제 유형 분류:**
- **회귀 문제 (Regression):** 연속적인 수치 예측 (예: 주택 가격, 매출 예측)
- **분류 문제 (Classification):** 범주형 결과 예측 (예: 이메일 스팸 판별, 이미지 분류)
- **클러스터링 (Clustering):** 유사한 데이터 그룹화 (예: 고객 세분화)
- **강화학습 (Reinforcement Learning):** 최적 행동 정책 학습 (예: 게임 AI, 추천 시스템)

### 1.3. 평가 지표 설정

프로젝트의 성공을 측정할 수 있는 **객관적이고 정량적인 지표**를 미리 정의해야 합니다:

**기술적 지표 예시:**
- 분류: 정확도(Accuracy), F1-score, ROC-AUC
- 회귀: RMSE, MAE, R-squared
- 클러스터링: Silhouette Score, Davies-Bouldin Index

**비즈니스 지표 예시:**
- 매출 증가율, 비용 절감액, 고객 만족도 개선 정도

---

## 2. 데이터 수집 (Data Collection)

### 2.1. 데이터 소스 식별

필요한 데이터의 종류와 출처를 파악하고 수집 전략을 수립합니다:

**데이터 소스 유형:**
- **내부 데이터:** 기업 데이터베이스, 로그 파일, CRM 시스템
- **외부 데이터:** 공개 데이터셋, API, 웹 크롤링
- **실시간 데이터:** 센서 데이터, 스트리밍 데이터
- **라벨링 데이터:** 수동으로 생성된 정답 데이터

### 2.2. 데이터 품질 고려사항

**데이터 품질 체크리스트:**
- **완정성(Completeness):** 누락된 데이터가 있는가?
- **정확성(Accuracy):** 데이터가 실제를 정확히 반영하는가?
- **일관성(Consistency):** 데이터 형식이 통일되어 있는가?
- **적시성(Timeliness):** 데이터가 최신 상태인가?
- **관련성(Relevance):** 해결하려는 문제와 관련이 있는가?

---

## 3. 탐색적 데이터 분석 및 전처리 (EDA & Preprocessing)

이 단계는 머신러닝 프로젝트에서 **가장 많은 시간(보통 전체의 60-80%)**을 차지하는 중요한 과정입니다.

### 3.1. 탐색적 데이터 분석 (Exploratory Data Analysis, EDA)

**기본 데이터 파악:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 기본 정보 확인
df.info()
df.describe()
df.head()

# 결측치 확인
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])

# 데이터 분포 시각화
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.select_dtypes(include=[np.number]).columns):
    plt.subplot(3, 3, i+1)
    df[column].hist(bins=30, alpha=0.7)
    plt.title(f'{column} Distribution')
    plt.xlabel(column)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
```

**상관관계 분석:**
```python
# 상관관계 히트맵
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# 타겟 변수와의 상관관계
target_correlation = df.corr()['target'].sort_values(ascending=False)
print("Target variable correlations:")
print(target_correlation)
```

### 3.2. 데이터 전처리 (Data Preprocessing)

**결측치 처리:**
```python
from sklearn.impute import SimpleImputer, KNNImputer

# 수치형 데이터: 평균/중앙값으로 대체
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

# 범주형 데이터: 최빈값으로 대체
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

# KNN 기반 결측치 대체 (고급)
knn_imputer = KNNImputer(n_neighbors=5)
df_imputed = knn_imputer.fit_transform(df)
```

**이상치 처리:**
```python
# IQR 방법으로 이상치 탐지
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# 이상치 범위 설정
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 이상치 제거 또는 변환
df_cleaned = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

# 또는 Winsorization (이상치를 임계값으로 변환)
from scipy.stats.mstats import winsorize
df['feature'] = winsorize(df['feature'], limits=[0.05, 0.05])
```

### 3.3. 특성 공학 (Feature Engineering)

**새로운 특성 생성:**
```python
# 날짜/시간 특성 추출
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month
df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek

# 수치형 특성의 변환
df['price_per_sqft'] = df['price'] / df['square_feet']
df['age'] = 2024 - df['year_built']

# 범주형 변수의 조합
df['location_type'] = df['city'] + '_' + df['property_type']

# 구간화 (Binning)
df['price_range'] = pd.cut(df['price'], 
                          bins=[0, 100000, 300000, 500000, float('inf')],
                          labels=['Low', 'Medium', 'High', 'Luxury'])
```

**특성 스케일링:**
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 표준화 (평균 0, 표준편차 1)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numeric_columns])

# 정규화 (0-1 범위)
minmax_scaler = MinMaxScaler()
df_normalized = minmax_scaler.fit_transform(df[numeric_columns])

# 로버스트 스케일링 (이상치에 강함)
robust_scaler = RobustScaler()
df_robust = robust_scaler.fit_transform(df[numeric_columns])
```

**범주형 변수 인코딩:**
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# Label Encoding (순서형 범주)
label_encoder = LabelEncoder()
df['education_level'] = label_encoder.fit_transform(df['education_level'])

# One-Hot Encoding (명목형 범주)
df_encoded = pd.get_dummies(df, columns=['city', 'category'], drop_first=True)

# 고카디널리티 범주형 변수 처리
from category_encoders import TargetEncoder
target_encoder = TargetEncoder()
df['city_encoded'] = target_encoder.fit_transform(df['city'], df['target'])
```

### 3.4. 데이터 분할 (Data Splitting)

```python
from sklearn.model_selection import train_test_split

# 기본 분할 (Train: 60%, Validation: 20%, Test: 20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")
```

---

## 4. 모델 선택 (Model Selection)

### 4.1. 문제 유형별 적합한 알고리즘

**회귀 문제:**
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# 다양한 회귀 모델 후보
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}
```

**분류 문제:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# 다양한 분류 모델 후보
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Naive Bayes': GaussianNB()
}
```

### 4.2. 베이스라인 모델 구축

```python
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# 분류 문제의 베이스라인
dummy_classifier = DummyClassifier(strategy='most_frequent')
dummy_classifier.fit(X_train, y_train)
dummy_pred = dummy_classifier.predict(X_val)
baseline_accuracy = accuracy_score(y_val, dummy_pred)

print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
print("모든 모델은 이 성능을 넘어야 의미가 있습니다.")
```

---

## 5. 모델 훈련 (Model Training)

### 5.1. 기본 모델 훈련

```python
from sklearn.metrics import classification_report, mean_squared_error, r2_score

# 모델 훈련 및 평가 함수
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    # 모델 훈련
    model.fit(X_train, y_train)
    
    # 예측
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # 평가 (분류의 경우)
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Overfitting Check: {train_acc - val_acc:.4f}")
    
    return model, val_pred

# 모든 모델 훈련 및 비교
results = {}
for name, model in models.items():
    trained_model, predictions = train_and_evaluate_model(
        model, X_train, y_train, X_val, y_val, name
    )
    results[name] = {
        'model': trained_model,
        'predictions': predictions
    }
```

### 5.2. 앙상블 기법

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Voting 앙상블
voting_classifier = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('svc', SVC(probability=True)),
        ('lr', LogisticRegression())
    ],
    voting='soft'  # 'hard' for majority voting
)

voting_classifier.fit(X_train, y_train)
voting_pred = voting_classifier.predict(X_val)
voting_acc = accuracy_score(y_val, voting_pred)

print(f"Voting Ensemble Accuracy: {voting_acc:.4f}")
```

---

## 6. 모델 평가 (Model Evaluation)

### 6.1. 분류 모델 평가

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt

# 혼동 행렬 (Confusion Matrix)
cm = confusion_matrix(y_val, val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# 분류 리포트
print("\nClassification Report:")
print(classification_report(y_val, val_pred))

# ROC Curve (이진 분류의 경우)
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_val, model.predict_proba(X_val)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

### 6.2. 회귀 모델 평가

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 회귀 평가 지표
mae = mean_absolute_error(y_val, val_pred)
mse = mean_squared_error(y_val, val_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, val_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# 실제값 vs 예측값 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_val, val_pred, alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# 잔차 분석 (Residual Analysis)
residuals = y_val - val_pred
plt.figure(figsize=(10, 6))
plt.scatter(val_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

---

## 7. 하이퍼파라미터 튜닝 (Hyperparameter Tuning)

### 7.1. Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Random Forest 하이퍼파라미터 튜닝 예시
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# 최적 모델로 예측
best_model = grid_search.best_estimator_
best_pred = best_model.predict(X_val)
best_accuracy = accuracy_score(y_val, best_pred)
print(f"Best model validation accuracy: {best_accuracy:.4f}")
```

### 7.2. Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 랜덤 서치를 위한 확률 분포 정의
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=100,  # 100번 무작위 샘플링
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
print(f"Best random search parameters: {random_search.best_params_}")
```

### 7.3. Bayesian Optimization (고급)

```python
# pip install scikit-optimize 필요
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Bayesian optimization을 위한 검색 공간 정의
search_spaces = {
    'n_estimators': Integer(50, 300),
    'max_depth': Categorical([5, 10, 15, 20, None]),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform')
}

bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=search_spaces,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

bayes_search.fit(X_train, y_train)
print(f"Best Bayesian optimization parameters: {bayes_search.best_params_}")
```

---

## 8. 모델 배포 및 모니터링 (Deployment & Monitoring)

### 8.1. 모델 저장 및 버전 관리

```python
import joblib
import pickle
from datetime import datetime

# 모델 저장
model_filename = f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
joblib.dump(best_model, model_filename)

# 전처리 파이프라인도 함께 저장
preprocessing_filename = f"preprocessor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
joblib.dump(scaler, preprocessing_filename)

# 모델 메타데이터 저장
model_metadata = {
    'model_type': 'RandomForestClassifier',
    'features': list(X_train.columns),
    'training_accuracy': train_acc,
    'validation_accuracy': val_acc,
    'hyperparameters': best_model.get_params(),
    'training_date': datetime.now().isoformat()
}

import json
with open(f"model_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
    json.dump(model_metadata, f, indent=2)
```

### 8.2. 모델 서빙을 위한 API 구축

```python
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# 모델과 전처리 파이프라인 로드
model = joblib.load('best_model_20241213_143022.pkl')
preprocessor = joblib.load('preprocessor_20241213_143022.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # JSON 데이터 받기
        data = request.get_json()
        
        # DataFrame으로 변환
        df = pd.DataFrame([data])
        
        # 전처리 적용
        df_processed = preprocessor.transform(df)
        
        # 예측 수행
        prediction = model.predict(df_processed)[0]
        probability = model.predict_proba(df_processed)[0].max()
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(probability),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### 8.3. 모델 성능 모니터링

```python
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class ModelMonitor:
    def __init__(self, model, threshold=0.05):
        self.model = model
        self.threshold = threshold
        self.performance_history = []
        self.data_drift_history = []
        
    def log_prediction(self, features, prediction, actual=None):
        """예측과 실제 결과를 로깅"""
        log_entry = {
            'timestamp': datetime.now(),
            'features': features,
            'prediction': prediction,
            'actual': actual
        }
        self.performance_history.append(log_entry)
        
    def calculate_data_drift(self, new_data, reference_data):
        """데이터 드리프트 감지 (단순 통계 기반)"""
        drift_scores = {}
        
        for column in reference_data.columns:
            if reference_data[column].dtype in ['int64', 'float64']:
                # 평균과 표준편차 비교
                ref_mean = reference_data[column].mean()
                new_mean = new_data[column].mean()
                ref_std = reference_data[column].std()
                
                # 정규화된 평균 차이
                if ref_std > 0:
                    drift_score = abs(new_mean - ref_mean) / ref_std
                    drift_scores[column] = drift_score
                    
        return drift_scores
        
    def check_model_performance(self, recent_days=7):
        """최근 성능 확인"""
        recent_data = [
            entry for entry in self.performance_history
            if entry['timestamp'] > datetime.now() - timedelta(days=recent_days)
            and entry['actual'] is not None
        ]
        
        if len(recent_data) < 10:
            print("성능 평가를 위한 충분한 데이터가 없습니다.")
            return None
            
        predictions = [entry['prediction'] for entry in recent_data]
        actuals = [entry['actual'] for entry in recent_data]
        
        accuracy = sum([p == a for p, a in zip(predictions, actuals)]) / len(predictions)
        
        print(f"최근 {recent_days}일 성능: {accuracy:.4f}")
        
        if accuracy < self.threshold:
            print("⚠️  성능 저하가 감지되었습니다. 모델 재학습을 검토하세요.")
            
        return accuracy

# 사용 예시
monitor = ModelMonitor(best_model, threshold=0.85)

# 새로운 예측 로깅
new_features = X_test.iloc[0:1]
prediction = model.predict(new_features)[0]
actual = y_test.iloc[0]

monitor.log_prediction(new_features.values[0], prediction, actual)
```

---

## 예제 및 풀이 (Examples and Solutions)

### 예제 1: 주택 가격 예측 프로젝트

**문제:** 부동산 회사에서 주택의 특성을 바탕으로 가격을 예측하는 시스템을 구축하려고 합니다. 다음 데이터가 주어졌을 때, 완전한 머신러닝 워크플로우를 구현해보세요.

**데이터 특성:**
- `sqft_living`: 거주 면적 (평방피트)
- `bedrooms`: 침실 개수
- `bathrooms`: 욕실 개수
- `floors`: 층 수
- `condition`: 집 상태 (1-5점)
- `grade`: 건축 등급 (1-13점)
- `yr_built`: 건축 연도
- `price`: 가격 (예측 대상)

**풀이:**

**1단계: 문제 정의**
```python
# 비즈니스 문제: 주택 가격을 정확히 예측하여 적정 가격 책정
# ML 문제: 회귀 문제 (연속적인 가격 예측)
# 성공 지표: RMSE < $50,000, R² > 0.85
```

**2단계: 데이터 수집 및 탐색**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 샘플 데이터 생성 (실제로는 데이터베이스에서 로드)
np.random.seed(42)
n_samples = 1000

data = {
    'sqft_living': np.random.normal(2000, 500, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.normal(2.5, 0.8, n_samples),
    'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples),
    'condition': np.random.randint(1, 6, n_samples),
    'grade': np.random.randint(3, 14, n_samples),
    'yr_built': np.random.randint(1950, 2024, n_samples)
}

# 가격 생성 (특성들과 선형 관계 + 노이즈)
price = (
    data['sqft_living'] * 150 +
    data['bedrooms'] * 10000 +
    data['bathrooms'] * 15000 +
    data['floors'] * 20000 +
    data['condition'] * 25000 +
    data['grade'] * 30000 +
    (data['yr_built'] - 1950) * 1000 +
    np.random.normal(0, 50000, n_samples)
)

data['price'] = np.abs(price)  # 가격은 양수
df = pd.DataFrame(data)

# 기본 데이터 탐색
print("=== 데이터 기본 정보 ===")
print(df.info())
print("\n=== 기술 통계 ===")
print(df.describe())

# 타겟 변수 분포 확인
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(df['price'], bins=30, alpha=0.7, color='skyblue')
plt.title('Price Distribution')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(np.log(df['price']), bins=30, alpha=0.7, color='lightcoral')
plt.title('Log(Price) Distribution')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
```

**3단계: 특성 공학 및 전처리**
```python
# 새로운 특성 생성
df['house_age'] = 2024 - df['yr_built']
df['price_per_sqft'] = df['price'] / df['sqft_living']
df['bedroom_bathroom_ratio'] = df['bedrooms'] / (df['bathrooms'] + 0.1)  # 0으로 나누기 방지

# 이상치 제거 (IQR 방법)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df_clean = remove_outliers(df, 'price')
print(f"이상치 제거 후 데이터 크기: {df_clean.shape[0]} (제거된 행: {df.shape[0] - df_clean.shape[0]})")

# 특성과 타겟 분리
features = ['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'condition', 'grade', 'house_age']
X = df_clean[features]
y = df_clean['price']

# 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"훈련 세트: {X_train.shape}")
print(f"검증 세트: {X_val.shape}")
print(f"테스트 세트: {X_test.shape}")

# 특성 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

**4단계: 모델 선택 및 훈련**
```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 여러 모델 정의
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# 모델 훈련 및 평가
results = {}

for name, model in models.items():
    # 훈련
    if 'Linear' in name or 'Ridge' in name:
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
    else:
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
    
    # 평가
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    results[name] = {
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'overfitting': train_rmse - val_rmse
    }
    
    print(f"\n=== {name} ===")
    print(f"Training RMSE: ${train_rmse:,.2f}")
    print(f"Validation RMSE: ${val_rmse:,.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Validation R²: {val_r2:.4f}")
    print(f"Overfitting (차이): ${train_rmse - val_rmse:,.2f}")

# 최적 모델 선택
best_model_name = min(results.keys(), key=lambda x: results[x]['val_rmse'])
print(f"\n🏆 최적 모델: {best_model_name}")
```

**5단계: 하이퍼파라미터 튜닝**
```python
from sklearn.model_selection import GridSearchCV

# Random Forest 튜닝 (예시)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    rf_model, param_grid, 
    cv=5, scoring='neg_root_mean_squared_error',
    n_jobs=-1, verbose=1
)

grid_search.fit(X_train, y_train)
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 CV 점수: ${-grid_search.best_score_:,.2f} RMSE")

# 최적 모델로 최종 예측
best_rf_model = grid_search.best_estimator_
test_pred = best_rf_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)

print(f"\n=== 최종 테스트 성능 ===")
print(f"Test RMSE: ${test_rmse:,.2f}")
print(f"Test R²: {test_r2:.4f}")

# 목표 달성 여부 확인
if test_rmse < 50000 and test_r2 > 0.85:
    print("✅ 목표 성능 달성!")
else:
    print("❌ 목표 성능 미달성. 추가 개선 필요.")
```

**6단계: 모델 해석**
```python
# 특성 중요도 시각화
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

print("=== 특성 중요도 순위 ===")
for i, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")
```

**7단계: 모델 배포 준비**
```python
import joblib

# 최종 모델과 전처리 파이프라인 저장
joblib.dump(best_rf_model, 'house_price_model.pkl')
joblib.dump(scaler, 'house_price_scaler.pkl')

# 예측 함수 작성
def predict_house_price(sqft_living, bedrooms, bathrooms, floors, condition, grade, house_age):
    """주택 가격 예측 함수"""
    # 입력 데이터 준비
    input_data = pd.DataFrame({
        'sqft_living': [sqft_living],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'floors': [floors],
        'condition': [condition],
        'grade': [grade],
        'house_age': [house_age]
    })
    
    # 예측 수행
    prediction = best_rf_model.predict(input_data)[0]
    
    return prediction

# 테스트 예측
sample_price = predict_house_price(2000, 3, 2.5, 2, 4, 8, 10)
print(f"예측된 주택 가격: ${sample_price:,.2f}")
```

**해설:**
이 예제는 완전한 머신러닝 워크플로우를 보여줍니다:
1. **문제 정의**: 회귀 문제로 명확히 정의하고 성공 지표 설정
2. **데이터 탐색**: EDA를 통한 데이터 이해
3. **특성 공학**: 새로운 특성 생성 및 전처리
4. **모델 선택**: 여러 모델 비교 평가
5. **최적화**: 하이퍼파라미터 튜닝으로 성능 개선
6. **해석**: 특성 중요도 분석으로 모델 이해
7. **배포 준비**: 실제 사용을 위한 함수 및 저장

각 단계는 다음 단계의 기초가 되며, 필요에 따라 이전 단계로 돌아가 개선하는 **반복적 과정**임을 보여줍니다.

---

## 핵심 요약 (Key Takeaways)

### 워크플로우의 특징
- **반복적이고 순환적**: 선형적 과정이 아닌 피드백을 통한 지속적 개선
- **데이터 중심**: 전체 시간의 60-80%가 데이터 관련 작업에 소요
- **실험적 접근**: 가설 설정 → 실험 → 평가 → 개선의 과학적 방법론
- **협업 필수**: 도메인 전문가, 데이터 과학자, 엔지니어 간 긴밀한 협조

### 성공 요인
1. **명확한 문제 정의**: 모호한 목표는 실패의 지름길
2. **양질의 데이터**: "Garbage in, Garbage out"
3. **적절한 평가**: 비즈니스 목표와 연결된 평가 지표
4. **지속적 모니터링**: 배포는 끝이 아닌 시작
5. **팀워크**: 다양한 역할의 전문가들과의 효과적 소통

### 일반적인 실수와 해결책
- **데이터 리키지**: 미래 정보가 모델에 포함되지 않도록 주의
- **과적합**: 교차 검증과 정규화 기법 활용
- **평가 편향**: 층화 추출과 적절한 검증 세트 구성
- **스케일링 누락**: 다양한 범위의 특성들에 대한 정규화
- **비즈니스 맥락 무시**: 기술적 성능만이 아닌 실제 가치 고려
