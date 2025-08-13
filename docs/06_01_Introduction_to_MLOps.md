# MLOps 개요 (Introduction to MLOps)

현대의 머신러닝 프로젝트는 더 이상 데이터 과학자가 개인 노트북에서 모델을 훈련하고 끝나는 것으로 완료되지 않습니다. 실제 비즈니스 가치를 창출하기 위해서는 모델을 **프로덕션 환경에 배포**하고, **지속적으로 모니터링**하며, **새로운 데이터에 맞춰 업데이트**하는 복잡한 과정을 거쳐야 합니다.

**MLOps(Machine Learning Operations)**는 이러한 도전과제를 해결하기 위한 **사람, 프로세스, 그리고 기술의 통합된 접근법**으로, 머신러닝 모델의 전체 생명주기를 체계적으로 관리하고 자동화하는 방법론입니다.

---

## 1. MLOps의 정의와 필요성

### 1.1. MLOps란 무엇인가?

**MLOps(Machine Learning Operations)**는 머신러닝과 DevOps의 결합으로, 다음과 같이 정의할 수 있습니다:

> **MLOps는 머신러닝 모델의 개발, 배포, 운영을 위한 일련의 관행, 도구, 그리고 문화로서, 머신러닝 시스템의 신뢰성, 확장성, 그리고 성능을 보장하기 위한 체계적인 접근법입니다.**

**핵심 구성요소:**
- **사람 (People)**: 데이터 사이언티스트, ML 엔지니어, 데이터 엔지니어, DevOps 엔지니어
- **프로세스 (Process)**: 모델 개발 워크플로우, 품질 관리, 배포 절차
- **기술 (Technology)**: 자동화 도구, 모니터링 시스템, 인프라

### 1.2. MLOps의 필요성

#### 1.2.1. 연구와 프로덕션 간의 간극

**연구 환경의 특징:**
```python
# 연구/실험 환경에서의 일반적인 코드
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 데이터 로딩
df = pd.read_csv('data.csv')

# 간단한 전처리
X = df.drop('target', axis=1).fillna(0)
y = df['target']

# 모델 훈련
model = RandomForestClassifier()
model.fit(X, y)

print(f"훈련 정확도: {model.score(X, y)}")
```

**프로덕션 환경에서 필요한 고려사항:**
```python
# 프로덕션 환경에서 필요한 코드 구조
import logging
import joblib
from datetime import datetime
from typing import Dict, Any
import pandas as pd
from sklearn.base import BaseEstimator
from monitoring import ModelMonitor
from data_validation import validate_input_data

class ProductionMLModel:
    def __init__(self, model_path: str, monitor: ModelMonitor):
        self.model = joblib.load(model_path)
        self.monitor = monitor
        self.version = "v1.2.3"
        self.logger = logging.getLogger(__name__)
        
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 입력 데이터 검증
            validated_data = validate_input_data(input_data)
            
            # 특성 추출
            features = self._extract_features(validated_data)
            
            # 예측 수행
            prediction = self.model.predict(features)[0]
            confidence = self.model.predict_proba(features)[0].max()
            
            # 모니터링 로깅
            self.monitor.log_prediction(features, prediction, confidence)
            
            # 결과 반환
            result = {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'model_version': self.version,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Prediction made: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def _extract_features(self, data: Dict[str, Any]) -> pd.DataFrame:
        # 프로덕션에서는 정교한 특성 추출 로직 필요
        pass
```

#### 1.2.2. 비즈니스 요구사항

**1) 신속한 모델 업데이트**
- 새로운 데이터에 따른 모델 성능 저하 대응
- 비즈니스 로직 변경에 따른 빠른 모델 조정
- A/B 테스트를 통한 모델 성능 비교

**2) 확장성과 안정성**
- 대용량 트래픽 처리 능력
- 시스템 장애 시 자동 복구
- 예측 지연 시간 최소화

**3) 규제 및 컴플라이언스**
- 모델 결정 과정의 투명성 (설명 가능한 AI)
- 데이터 프라이버시 보호 (GDPR, CCPA)
- 감사 추적 가능성

#### 1.2.3. 기술적 도전과제

**모델 드리프트 문제:**
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 데이터 드리프트 시뮬레이션
def simulate_data_drift():
    # 훈련 시기 데이터 분포
    train_data = np.random.normal(0, 1, 1000)
    
    # 6개월 후 데이터 분포 (드리프트 발생)
    production_data = np.random.normal(0.5, 1.2, 1000)  # 평균과 분산 변화
    
    # KS 검정으로 분포 변화 감지
    ks_statistic, p_value = stats.ks_2samp(train_data, production_data)
    
    print(f"KS Statistic: {ks_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("⚠️  데이터 드리프트 감지됨! 모델 재학습 필요")
    else:
        print("✅ 데이터 분포 안정")
    
    # 시각화
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(train_data, alpha=0.7, label='Training Data', bins=30)
    plt.hist(production_data, alpha=0.7, label='Production Data', bins=30)
    plt.legend()
    plt.title('Data Distribution Comparison')
    
    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(train_data[:100]), label='Training Cumulative')
    plt.plot(np.cumsum(production_data[:100]), label='Production Cumulative')
    plt.legend()
    plt.title('Cumulative Values Over Time')
    plt.show()

simulate_data_drift()
```

**모델 성능 모니터링:**
```python
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime, timedelta

class ModelPerformanceTracker:
    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.timestamps = []
        
    def log_prediction(self, prediction, actual=None, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
            
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(timestamp)
        
    def calculate_performance_metrics(self, window_days=7):
        # 최근 N일간의 성능 계산
        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_indices = [i for i, ts in enumerate(self.timestamps) if ts > cutoff_date]
        
        if not recent_indices:
            return None
            
        recent_predictions = [self.predictions[i] for i in recent_indices]
        recent_actuals = [self.actuals[i] for i in recent_indices if self.actuals[i] is not None]
        
        if len(recent_actuals) < len(recent_predictions) * 0.5:  # 50% 이상 실제값 필요
            print("⚠️  성능 평가를 위한 충분한 실제값 없음")
            return None
            
        # 성능 지표 계산
        accuracy = accuracy_score(recent_actuals, recent_predictions[:len(recent_actuals)])
        f1 = f1_score(recent_actuals, recent_predictions[:len(recent_actuals)], average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'sample_size': len(recent_actuals),
            'period': f'Last {window_days} days'
        }
    
    def detect_performance_degradation(self, baseline_accuracy=0.85, threshold=0.05):
        current_metrics = self.calculate_performance_metrics()
        
        if current_metrics is None:
            return False
            
        performance_drop = baseline_accuracy - current_metrics['accuracy']
        
        if performance_drop > threshold:
            print(f"🚨 성능 저하 감지: {performance_drop:.3f} 포인트 하락")
            print(f"현재 정확도: {current_metrics['accuracy']:.3f}")
            print(f"기준 정확도: {baseline_accuracy:.3f}")
            return True
        else:
            print(f"✅ 성능 안정: 현재 정확도 {current_metrics['accuracy']:.3f}")
            return False

# 사용 예시
tracker = ModelPerformanceTracker()

# 예측 로깅 시뮬레이션
for i in range(100):
    prediction = np.random.choice([0, 1])
    # 시간이 지날수록 성능이 저하되는 상황 시뮬레이션
    accuracy_decay = max(0.5, 0.9 - i * 0.005)
    actual = prediction if np.random.random() < accuracy_decay else 1 - prediction
    
    tracker.log_prediction(prediction, actual)

# 성능 저하 감지
tracker.detect_performance_degradation()
```

---

## 2. DevOps와 MLOps의 차이점

### 2.1. 핵심 차이점 비교

| 측면 | DevOps | MLOps |
|------|--------|-------|
| **주요 아티팩트** | 코드, 설정 파일 | 코드 + 데이터 + 모델 + 하이퍼파라미터 |
| **변경 트리거** | 코드 변경 | 코드 변경 + 데이터 변경 + 성능 저하 |
| **테스트 종류** | 단위/통합/UI 테스트 | 데이터 테스트 + 모델 테스트 + 성능 테스트 |
| **배포 복잡성** | 중간 | 높음 (데이터 의존성, 모델 가중치) |
| **모니터링 범위** | 시스템 메트릭 | 시스템 + 데이터 + 모델 성능 메트릭 |
| **롤백 복잡성** | 간단 (코드 버전만 관리) | 복잡 (코드 + 데이터 + 모델 버전 관리) |
| **예측 가능성** | 높음 (결정론적) | 낮음 (확률적, 데이터 의존적) |

### 2.2. MLOps만의 고유한 요소들

#### 2.2.1. 데이터 버전 관리 (Data Versioning)

**DVC(Data Version Control) 사용 예시:**
```bash
# DVC 초기화
dvc init

# 데이터 추가 및 버전 관리
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "Add raw dataset v1.0"

# 데이터 변경 후 새 버전 생성
dvc add data/processed/features_v2.csv
git add data/processed/features_v2.csv.dvc
git commit -m "Add processed features v2.0"

# 특정 데이터 버전으로 되돌리기
git checkout data-v1.0
dvc checkout
```

#### 2.2.2. 실험 추적 (Experiment Tracking)

**MLflow를 사용한 실험 관리:**
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# MLflow 실험 시작
mlflow.set_experiment("house_price_prediction")

def train_and_log_model(n_estimators, max_depth, min_samples_split, X_train, y_train, X_test, y_test):
    with mlflow.start_run():
        # 하이퍼파라미터 로깅
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        
        # 모델 훈련
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # 예측 및 평가
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        
        # 메트릭 로깅
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("overfitting", train_accuracy - test_accuracy)
        
        # 모델 저장
        mlflow.sklearn.log_model(model, "model")
        
        # 특성 중요도 시각화 및 로깅
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        feature_importance = model.feature_importances_
        plt.barh(range(len(feature_importance)), feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        mlflow.log_artifact('feature_importance.png')
        
        return model

# 다양한 하이퍼파라미터로 실험 수행
experiments = [
    {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 2},
    {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
    {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 10}
]

# 실제 데이터로 실험 (예시용 더미 데이터)
X_train, X_test = np.random.randn(1000, 10), np.random.randn(200, 10)
y_train, y_test = np.random.randint(0, 2, 1000), np.random.randint(0, 2, 200)

for exp in experiments:
    print(f"실험 진행: {exp}")
    train_and_log_model(X_train=X_train, y_train=y_train, 
                       X_test=X_test, y_test=y_test, **exp)
```

#### 2.2.3. 모델 서빙 아키텍처

**REST API 기반 모델 서빙:**
```python
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging

app = Flask(__name__)

class ModelServer:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.version = "1.0.0"
        self.load_time = datetime.now()
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 성능 메트릭
        self.prediction_count = 0
        self.total_inference_time = 0.0
        
    def predict(self, features):
        start_time = datetime.now()
        
        try:
            # 입력 검증
            if not isinstance(features, (list, np.ndarray)):
                raise ValueError("Features must be a list or numpy array")
            
            # 예측 수행
            features_array = np.array(features).reshape(1, -1)
            prediction = self.model.predict(features_array)[0]
            probability = self.model.predict_proba(features_array)[0]
            
            # 성능 메트릭 업데이트
            inference_time = (datetime.now() - start_time).total_seconds()
            self.prediction_count += 1
            self.total_inference_time += inference_time
            
            result = {
                'prediction': int(prediction),
                'probability': probability.tolist(),
                'model_version': self.version,
                'inference_time_ms': inference_time * 1000,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Prediction completed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    def get_health_status(self):
        avg_inference_time = (self.total_inference_time / self.prediction_count 
                             if self.prediction_count > 0 else 0)
        
        return {
            'status': 'healthy',
            'model_version': self.version,
            'uptime': str(datetime.now() - self.load_time),
            'prediction_count': self.prediction_count,
            'average_inference_time_ms': avg_inference_time * 1000
        }

# 모델 서버 인스턴스 생성
model_server = ModelServer('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')
        
        if features is None:
            return jsonify({'error': 'Features not provided'}), 400
        
        result = model_server.predict(features)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify(model_server.get_health_status())

@app.route('/metrics', methods=['GET'])
def metrics():
    # Prometheus 형식의 메트릭 제공
    metrics = f"""# HELP model_predictions_total Total number of predictions made
# TYPE model_predictions_total counter
model_predictions_total {model_server.prediction_count}

# HELP model_inference_duration_seconds Time spent on inference
# TYPE model_inference_duration_seconds summary
model_inference_duration_seconds_sum {model_server.total_inference_time}
model_inference_duration_seconds_count {model_server.prediction_count}
"""
    return metrics, 200, {'Content-Type': 'text/plain'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

---

## 3. MLOps 성숙도 모델

MLOps 구현은 단계적으로 발전하며, Google에서 제안한 성숙도 모델을 기준으로 조직의 MLOps 수준을 평가할 수 있습니다.

### 3.1. Level 0: Manual Process (수동 프로세스)

**특징:**
- 데이터 과학자가 수동으로 모든 작업 수행
- Jupyter 노트북 중심의 실험적 접근
- 모델 배포는 일회성, 수동 작업

**구현 예시:**
```python
# Level 0 - 전형적인 수동 워크플로우
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. 데이터 로딩 (수동)
print("데이터 로딩...")
data = pd.read_csv('data.csv')

# 2. 전처리 (수동)
print("전처리 중...")
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. 모델 훈련 (수동)
print("모델 훈련 중...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. 평가 (수동)
score = model.score(X_test, y_test)
print(f"정확도: {score:.3f}")

# 5. 모델 저장 (수동)
joblib.dump(model, 'model_v1.pkl')
print("모델 저장 완료")

# 배포는 수동으로 파일 복사...
```

### 3.2. Level 1: ML Pipeline Automation (ML 파이프라인 자동화)

**특징:**
- 모델 훈련 파이프라인 자동화
- 지속적 훈련 (Continuous Training, CT)
- 실험 추적 및 모델 레지스트리 도입

**구현 예시:**
```python
# Level 1 - 자동화된 ML 파이프라인
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import mlflow
import mlflow.sklearn

class MLTrainingPipeline:
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)
        
    def create_pipeline(self):
        pipeline_options = PipelineOptions()
        
        with beam.Pipeline(options=pipeline_options) as pipeline:
            # 데이터 로딩 및 전처리
            (pipeline
             | 'ReadData' >> beam.io.ReadFromText('gs://bucket/data/*.csv')
             | 'ParseCSV' >> beam.Map(self.parse_csv)
             | 'Preprocess' >> beam.Map(self.preprocess)
             | 'TrainModel' >> beam.Map(self.train_model)
             | 'EvaluateModel' >> beam.Map(self.evaluate_model)
             | 'RegisterModel' >> beam.Map(self.register_model))
    
    def parse_csv(self, line):
        # CSV 파싱 로직
        pass
    
    def preprocess(self, data):
        # 전처리 로직
        pass
    
    def train_model(self, processed_data):
        with mlflow.start_run():
            # 모델 훈련 로직
            model = RandomForestClassifier()
            # ... 훈련 코드 ...
            mlflow.sklearn.log_model(model, "model")
            return model
    
    def evaluate_model(self, model):
        # 평가 로직
        pass
    
    def register_model(self, model):
        # 모델 레지스트리에 등록
        pass

# 파이프라인 실행
pipeline = MLTrainingPipeline("automated_training")
pipeline.create_pipeline()
```

### 3.3. Level 2: CI/CD Pipeline Automation (CI/CD 파이프라인 자동화)

**특징:**
- 완전한 CI/CD/CT 구현
- 자동화된 테스트 및 검증
- 프로덕션 배포 자동화

**GitHub Actions를 활용한 CI/CD 예시:**
```yaml
# .github/workflows/mlops-pipeline.yml
name: MLOps Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 0'  # 매주 일요일 2시에 재훈련

jobs:
  data-validation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Data Quality Tests
      run: |
        python tests/test_data_quality.py
        python tests/test_data_schema.py
    
    - name: Data Drift Detection
      run: |
        python src/data_drift_detection.py
  
  model-training:
    needs: data-validation
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Train Model
      run: |
        python src/train_model.py
    
    - name: Model Validation Tests
      run: |
        python tests/test_model_performance.py
        python tests/test_model_bias.py
    
    - name: Upload Model Artifacts
      uses: actions/upload-artifact@v2
      with:
        name: trained-model
        path: models/
  
  model-deployment:
    needs: model-training
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Download Model
      uses: actions/download-artifact@v2
      with:
        name: trained-model
        path: models/
    
    - name: Deploy to Staging
      run: |
        docker build -t model-server:latest .
        docker tag model-server:latest registry.com/model-server:staging
        docker push registry.com/model-server:staging
    
    - name: Integration Tests
      run: |
        python tests/test_api_integration.py
    
    - name: Deploy to Production
      if: success()
      run: |
        kubectl apply -f k8s/deployment.yaml
        kubectl rollout status deployment/model-server
```

**자동화된 모델 테스트:**
```python
# tests/test_model_performance.py
import unittest
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class TestModelPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load('models/latest_model.pkl')
        cls.test_data = pd.read_csv('data/test_set.csv')
        cls.X_test = cls.test_data.drop('target', axis=1)
        cls.y_test = cls.test_data['target']
    
    def test_minimum_accuracy(self):
        """모델이 최소 정확도를 만족하는지 테스트"""
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        
        minimum_accuracy = 0.85
        self.assertGreaterEqual(
            accuracy, minimum_accuracy,
            f"모델 정확도 {accuracy:.3f}가 최소 기준 {minimum_accuracy}를 만족하지 않음"
        )
    
    def test_f1_score(self):
        """F1 점수가 기준을 만족하는지 테스트"""
        predictions = self.model.predict(self.X_test)
        f1 = f1_score(self.y_test, predictions, average='weighted')
        
        minimum_f1 = 0.80
        self.assertGreaterEqual(
            f1, minimum_f1,
            f"F1 점수 {f1:.3f}가 최소 기준 {minimum_f1}를 만족하지 않음"
        )
    
    def test_prediction_time(self):
        """예측 시간이 허용 범위 내인지 테스트"""
        import time
        
        start_time = time.time()
        self.model.predict(self.X_test[:100])  # 100개 샘플 예측
        prediction_time = time.time() - start_time
        
        max_time_per_sample = 0.01  # 10ms per sample
        avg_time = prediction_time / 100
        
        self.assertLessEqual(
            avg_time, max_time_per_sample,
            f"평균 예측 시간 {avg_time*1000:.2f}ms가 기준 {max_time_per_sample*1000}ms를 초과"
        )
    
    def test_model_bias(self):
        """모델 편향성 테스트"""
        # 성별 등 민감한 속성에 대한 공정성 검증
        if 'gender' in self.X_test.columns:
            male_indices = self.X_test['gender'] == 'M'
            female_indices = self.X_test['gender'] == 'F'
            
            male_predictions = self.model.predict(self.X_test[male_indices])
            female_predictions = self.model.predict(self.X_test[female_indices])
            
            male_positive_rate = np.mean(male_predictions)
            female_positive_rate = np.mean(female_predictions)
            
            bias_threshold = 0.1  # 10% 차이까지 허용
            bias = abs(male_positive_rate - female_positive_rate)
            
            self.assertLessEqual(
                bias, bias_threshold,
                f"성별 간 편향 {bias:.3f}이 허용 기준 {bias_threshold}를 초과"
            )

if __name__ == '__main__':
    unittest.main()
```

---

## 4. MLOps 도구 및 기술 스택

### 4.1. MLOps 도구 생태계

**카테고리별 주요 도구들:**

| 카테고리 | 도구 | 설명 | 사용 예시 |
|----------|------|------|----------|
| **데이터 버전 관리** | DVC, Pachyderm | 데이터와 모델의 버전 관리 | 데이터셋 변경 이력 추적 |
| **실험 추적** | MLflow, Weights & Biases, Neptune | 실험 로깅 및 비교 | 하이퍼파라미터 튜닝 결과 관리 |
| **모델 레지스트리** | MLflow Model Registry, Seldon Core | 모델 저장 및 버전 관리 | 프로덕션 모델 관리 |
| **파이프라인 오케스트레이션** | Kubeflow, Apache Airflow, Prefect | 워크플로우 자동화 | 데이터 처리 → 훈련 → 배포 |
| **모델 서빙** | TensorFlow Serving, Seldon, BentoML | 모델 API 서비스 | REST API로 예측 제공 |
| **모니터링** | Prometheus, Grafana, Evidently | 시스템 및 모델 성능 모니터링 | 성능 저하 알림 |
| **인프라** | Kubernetes, Docker, AWS SageMaker | 컨테이너화 및 클라우드 배포 | 확장 가능한 모델 서빙 |

### 4.2. 통합된 MLOps 플랫폼 구축

**Docker를 활용한 컨테이너화:**
```dockerfile
# Dockerfile
FROM python:3.8-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# 헬스체크 스크립트
COPY healthcheck.py .

# 포트 노출
EXPOSE 5000

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python healthcheck.py

# 애플리케이션 실행
CMD ["python", "src/model_server.py"]
```

**Kubernetes 배포 설정:**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-server
  labels:
    app: ml-model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model-server
  template:
    metadata:
      labels:
        app: ml-model-server
    spec:
      containers:
      - name: model-server
        image: your-registry/ml-model-server:latest
        ports:
        - containerPort: 5000
        env:
        - name: MODEL_PATH
          value: "/app/models/latest_model.pkl"
        - name: MONITORING_ENDPOINT
          value: "http://prometheus:9090"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model-server
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 5. MLOps 파이프라인 구축 실습

### 5.1. 종단간 MLOps 파이프라인 예제

**프로젝트 구조:**
```
mlops-project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── schemas/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── drift_detection.py
│   └── api/
│       ├── __init__.py
│       └── server.py
├── tests/
│   ├── test_data_quality.py
│   ├── test_model_performance.py
│   └── test_api.py
├── config/
│   ├── config.yaml
│   └── model_config.yaml
├── notebooks/
│   └── exploratory_analysis.ipynb
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml
├── requirements.txt
├── setup.py
└── README.md
```

**데이터 수집 및 전처리:**
```python
# src/data/ingestion.py
import pandas as pd
import numpy as np
from datetime import datetime
import yaml
import logging

class DataIngestion:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = logging.getLogger(__name__)
        
    def collect_data(self, source_type='database'):
        """다양한 소스에서 데이터 수집"""
        if source_type == 'database':
            return self._collect_from_database()
        elif source_type == 'api':
            return self._collect_from_api()
        elif source_type == 'file':
            return self._collect_from_file()
        else:
            raise ValueError(f"지원하지 않는 소스 타입: {source_type}")
    
    def _collect_from_database(self):
        # 데이터베이스 연결 및 데이터 수집
        import sqlalchemy
        
        engine = sqlalchemy.create_engine(self.config['database']['connection_string'])
        query = self.config['database']['query']
        
        df = pd.read_sql(query, engine)
        self.logger.info(f"데이터베이스에서 {len(df)}개 행 수집")
        
        return df
    
    def _collect_from_api(self):
        # API에서 데이터 수집
        import requests
        
        api_url = self.config['api']['url']
        headers = self.config['api']['headers']
        
        response = requests.get(api_url, headers=headers)
        data = response.json()
        
        df = pd.DataFrame(data)
        self.logger.info(f"API에서 {len(df)}개 행 수집")
        
        return df
    
    def _collect_from_file(self):
        # 파일에서 데이터 수집
        file_path = self.config['file']['path']
        df = pd.read_csv(file_path)
        self.logger.info(f"파일에서 {len(df)}개 행 수집")
        
        return df
    
    def validate_data_schema(self, df):
        """데이터 스키마 검증"""
        expected_columns = self.config['schema']['columns']
        missing_columns = set(expected_columns) - set(df.columns)
        
        if missing_columns:
            raise ValueError(f"필수 컬럼 누락: {missing_columns}")
        
        # 데이터 타입 검증
        for col, expected_type in self.config['schema']['types'].items():
            if col in df.columns:
                if not df[col].dtype.name.startswith(expected_type):
                    self.logger.warning(f"컬럼 {col}의 타입이 예상과 다름: {df[col].dtype} vs {expected_type}")
        
        return True
    
    def check_data_quality(self, df):
        """데이터 품질 검사"""
        quality_report = {}
        
        # 결측치 비율
        missing_ratio = df.isnull().sum() / len(df)
        quality_report['missing_ratio'] = missing_ratio.to_dict()
        
        # 중복 행 비율
        duplicate_ratio = df.duplicated().sum() / len(df)
        quality_report['duplicate_ratio'] = duplicate_ratio
        
        # 수치형 컬럼의 이상치 검사
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | 
                           (df[col] > (Q3 + 1.5 * IQR))).sum()
            outliers[col] = outlier_count / len(df)
        
        quality_report['outlier_ratio'] = outliers
        
        self.logger.info(f"데이터 품질 보고서: {quality_report}")
        return quality_report
```

**모델 훈련 파이프라인:**
```python
# src/models/train.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        mlflow.set_experiment(config['experiment_name'])
        
    def train_model(self, X, y, model_params=None):
        """모델 훈련 및 MLflow 로깅"""
        with mlflow.start_run():
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 기본 파라미터 설정
            if model_params is None:
                model_params = self.config['model']['default_params']
            
            # 모델 초기화
            model = RandomForestClassifier(**model_params, random_state=42)
            
            # 교차 검증
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
            
            # 모델 훈련
            model.fit(X_train, y_train)
            
            # 예측 및 평가
            y_pred = model.predict(X_test)
            
            # MLflow에 파라미터 및 메트릭 로깅
            mlflow.log_params(model_params)
            mlflow.log_metric("cv_mean_f1", cv_scores.mean())
            mlflow.log_metric("cv_std_f1", cv_scores.std())
            mlflow.log_metric("test_f1", self._calculate_f1(y_test, y_pred))
            mlflow.log_metric("test_accuracy", self._calculate_accuracy(y_test, y_pred))
            
            # 혼동 행렬 시각화
            self._log_confusion_matrix(y_test, y_pred)
            
            # 특성 중요도 로깅
            self._log_feature_importance(model, X.columns)
            
            # 모델 저장
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=self.config['model']['registered_name']
            )
            
            # 로컬에도 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/model_{timestamp}.pkl"
            joblib.dump(model, model_path)
            
            return model, {
                'cv_mean_f1': cv_scores.mean(),
                'cv_std_f1': cv_scores.std(),
                'test_f1': self._calculate_f1(y_test, y_pred),
                'test_accuracy': self._calculate_accuracy(y_test, y_pred)
            }
    
    def _calculate_f1(self, y_true, y_pred):
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average='weighted')
    
    def _calculate_accuracy(self, y_true, y_pred):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)
    
    def _log_confusion_matrix(self, y_true, y_pred):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()
    
    def _log_feature_importance(self, model, feature_names):
        import matplotlib.pyplot as plt
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'][:20], importance_df['importance'][:20])
        plt.title('Top 20 Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        mlflow.log_artifact('feature_importance.png')
        plt.close()
        
        # CSV로도 저장
        importance_df.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')
```

**모니터링 시스템:**
```python
# src/monitoring/drift_detection.py
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import logging

class DataDriftDetector:
    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.logger = logging.getLogger(__name__)
        
    def detect_drift(self, current_data, significance_level=0.05):
        """데이터 드리프트 감지"""
        drift_results = {}
        
        numeric_columns = self.reference_data.select_dtypes(include=[np.number]).columns
        categorical_columns = self.reference_data.select_dtypes(include=['object']).columns
        
        # 수치형 컬럼 드리프트 검사 (KS Test)
        for col in numeric_columns:
            if col in current_data.columns:
                ks_stat, p_value = stats.ks_2samp(
                    self.reference_data[col].dropna(),
                    current_data[col].dropna()
                )
                
                drift_results[col] = {
                    'test': 'ks_test',
                    'statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': p_value < significance_level
                }
        
        # 범주형 컬럼 드리프트 검사 (Chi-square Test)
        for col in categorical_columns:
            if col in current_data.columns:
                ref_counts = self.reference_data[col].value_counts()
                cur_counts = current_data[col].value_counts()
                
                # 공통 카테고리만 비교
                common_categories = set(ref_counts.index) & set(cur_counts.index)
                
                if len(common_categories) > 1:
                    ref_freq = ref_counts[common_categories]
                    cur_freq = cur_counts[common_categories]
                    
                    # 빈도를 비율로 변환
                    ref_prop = ref_freq / ref_freq.sum()
                    cur_prop = cur_freq / cur_freq.sum()
                    
                    chi2_stat, p_value = stats.chisquare(cur_prop, ref_prop)
                    
                    drift_results[col] = {
                        'test': 'chi2_test',
                        'statistic': chi2_stat,
                        'p_value': p_value,
                        'drift_detected': p_value < significance_level
                    }
        
        # 전체 드리프트 요약
        total_features = len(drift_results)
        drifted_features = sum([r['drift_detected'] for r in drift_results.values()])
        drift_ratio = drifted_features / total_features if total_features > 0 else 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_features': total_features,
            'drifted_features': drifted_features,
            'drift_ratio': drift_ratio,
            'results': drift_results
        }
        
        self.logger.info(f"드리프트 검사 완료: {drifted_features}/{total_features} 특성에서 드리프트 감지")
        
        return summary
    
    def generate_drift_report(self, drift_results):
        """드리프트 보고서 생성"""
        import matplotlib.pyplot as plt
        
        drifted_features = [
            feature for feature, result in drift_results['results'].items()
            if result['drift_detected']
        ]
        
        if drifted_features:
            fig, axes = plt.subplots(len(drifted_features), 2, 
                                   figsize=(15, 5 * len(drifted_features)))
            
            if len(drifted_features) == 1:
                axes = axes.reshape(1, -1)
            
            for i, feature in enumerate(drifted_features):
                # 분포 비교
                axes[i, 0].hist(self.reference_data[feature].dropna(), 
                               alpha=0.7, label='Reference', bins=30)
                axes[i, 0].hist(self.current_data[feature].dropna(), 
                               alpha=0.7, label='Current', bins=30)
                axes[i, 0].set_title(f'{feature} - Distribution Comparison')
                axes[i, 0].legend()
                
                # P-value 시각화
                p_value = drift_results['results'][feature]['p_value']
                axes[i, 1].bar(['P-value'], [p_value], color='red' if p_value < 0.05 else 'green')
                axes[i, 1].axhline(y=0.05, color='r', linestyle='--', label='Significance Level')
                axes[i, 1].set_title(f'{feature} - P-value: {p_value:.4f}')
                axes[i, 1].legend()
            
            plt.tight_layout()
            plt.savefig('drift_report.png')
            plt.close()
            
            return 'drift_report.png'
        
        return None
```

**API 서버:**
```python
# src/api/server.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import yaml
from src.monitoring.drift_detection import DataDriftDetector

app = Flask(__name__)

class MLOpsAPIServer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = joblib.load(self.config['model']['path'])
        self.model_version = self.config['model']['version']
        self.reference_data = pd.read_csv(self.config['reference_data_path'])
        
        self.drift_detector = DataDriftDetector(self.reference_data)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 성능 메트릭
        self.prediction_count = 0
        self.total_inference_time = 0.0
        self.predictions_log = []
        
    def predict(self, features):
        start_time = datetime.now()
        
        try:
            # 입력 검증 및 전처리
            features_df = pd.DataFrame([features])
            
            # 드리프트 감지 (선택적)
            if len(self.predictions_log) % 100 == 0:  # 100번마다 드리프트 체크
                recent_data = pd.DataFrame(self.predictions_log[-100:])
                if not recent_data.empty:
                    drift_results = self.drift_detector.detect_drift(recent_data)
                    if drift_results['drift_ratio'] > 0.3:  # 30% 이상 특성에서 드리프트
                        self.logger.warning("상당한 데이터 드리프트가 감지됨!")
            
            # 예측 수행
            prediction = self.model.predict(features_df)[0]
            probabilities = self.model.predict_proba(features_df)[0]
            
            # 성능 메트릭 업데이트
            inference_time = (datetime.now() - start_time).total_seconds()
            self.prediction_count += 1
            self.total_inference_time += inference_time
            
            # 예측 로깅
            prediction_log = features.copy()
            prediction_log['prediction'] = prediction
            prediction_log['timestamp'] = start_time
            self.predictions_log.append(prediction_log)
            
            # 최근 1000개만 유지
            if len(self.predictions_log) > 1000:
                self.predictions_log = self.predictions_log[-1000:]
            
            result = {
                'prediction': int(prediction),
                'probabilities': probabilities.tolist(),
                'model_version': self.model_version,
                'inference_time_ms': inference_time * 1000,
                'timestamp': start_time.isoformat()
            }
            
            self.logger.info(f"예측 완료: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"예측 실패: {str(e)}")
            raise
    
    def get_health_status(self):
        avg_inference_time = (self.total_inference_time / self.prediction_count 
                             if self.prediction_count > 0 else 0)
        
        return {
            'status': 'healthy',
            'model_version': self.model_version,
            'prediction_count': self.prediction_count,
            'average_inference_time_ms': avg_inference_time * 1000,
            'last_prediction': self.predictions_log[-1]['timestamp'].isoformat() if self.predictions_log else None
        }
    
    def get_model_metrics(self):
        if not self.predictions_log:
            return {'message': '예측 데이터 없음'}
        
        recent_predictions = self.predictions_log[-100:]  # 최근 100개
        prediction_distribution = pd.Series([p['prediction'] for p in recent_predictions]).value_counts()
        
        return {
            'recent_prediction_count': len(recent_predictions),
            'prediction_distribution': prediction_distribution.to_dict(),
            'average_inference_time_ms': self.total_inference_time * 1000 / self.prediction_count
        }

# API 서버 인스턴스
ml_server = MLOpsAPIServer('config/config.yaml')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')
        
        if features is None:
            return jsonify({'error': 'Features not provided'}), 400
        
        result = ml_server.predict(features)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify(ml_server.get_health_status())

@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify(ml_server.get_model_metrics())

@app.route('/drift-check', methods=['POST'])
def drift_check():
    try:
        data = request.get_json()
        current_data = pd.DataFrame(data.get('data'))
        
        drift_results = ml_server.drift_detector.detect_drift(current_data)
        return jsonify(drift_results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

---

## 예제 및 풀이 (Examples and Solutions)

### 예제 1: 실시간 추천 시스템의 MLOps 구축

**문제:** 온라인 쇼핑몰의 실시간 상품 추천 시스템을 위한 MLOps 파이프라인을 구축하세요. 다음 요구사항을 만족해야 합니다:

**요구사항:**
1. 실시간 사용자 행동 데이터 처리 (클릭, 구매, 평점)
2. 일일 모델 재학습 및 성능 검증
3. A/B 테스트를 통한 모델 성능 비교
4. 추천 성능 실시간 모니터링
5. 레이턴시 < 100ms 보장

**풀이:**

**1단계: 아키텍처 설계**
```python
# 전체 시스템 아키텍처 설계
class RecommendationMLOpsArchitecture:
    """
    실시간 추천 시스템 MLOps 아키텍처
    
    Components:
    1. Data Streaming: Kafka + Spark Streaming
    2. Feature Store: Redis + Cassandra
    3. Model Training: Apache Airflow + MLflow
    4. Model Serving: TensorFlow Serving + Load Balancer
    5. Monitoring: Prometheus + Grafana + Custom Metrics
    6. A/B Testing: Custom Framework
    """
    
    def __init__(self):
        self.components = {
            'data_streaming': 'Kafka + Spark',
            'feature_store': 'Redis + Cassandra',
            'model_training': 'Airflow + MLflow',
            'model_serving': 'TF Serving',
            'monitoring': 'Prometheus + Grafana',
            'ab_testing': 'Custom Framework'
        }
```

**2단계: 실시간 데이터 처리**
```python
# 실시간 데이터 처리 파이프라인
from kafka import KafkaConsumer, KafkaProducer
import json
import redis
from datetime import datetime
import pandas as pd

class RealtimeDataProcessor:
    def __init__(self):
        self.consumer = KafkaConsumer(
            'user-events',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
    
    def process_user_events(self):
        """사용자 이벤트 실시간 처리"""
        for message in self.consumer:
            event = message.value
            
            # 이벤트 타입별 처리
            if event['event_type'] == 'click':
                self.process_click_event(event)
            elif event['event_type'] == 'purchase':
                self.process_purchase_event(event)
            elif event['event_type'] == 'rating':
                self.process_rating_event(event)
            
            # 실시간 특성 업데이트
            self.update_user_features(event['user_id'], event)
            
            # 모델 성능 모니터링을 위한 데이터 전송
            self.send_monitoring_data(event)
    
    def process_click_event(self, event):
        """클릭 이벤트 처리"""
        user_id = event['user_id']
        item_id = event['item_id']
        timestamp = event['timestamp']
        
        # 사용자 클릭 히스토리 업데이트
        click_key = f"user:{user_id}:clicks"
        self.redis_client.lpush(click_key, json.dumps({
            'item_id': item_id,
            'timestamp': timestamp
        }))
        self.redis_client.ltrim(click_key, 0, 99)  # 최근 100개만 유지
        
        # 아이템 인기도 업데이트
        popularity_key = f"item:{item_id}:popularity"
        self.redis_client.incr(popularity_key)
        
        # CTR 계산을 위한 노출 수 증가
        impression_key = f"item:{item_id}:impressions"
        self.redis_client.incr(impression_key)
    
    def process_purchase_event(self, event):
        """구매 이벤트 처리"""
        user_id = event['user_id']
        item_id = event['item_id']
        
        # 구매 히스토리 업데이트
        purchase_key = f"user:{user_id}:purchases"
        self.redis_client.lpush(purchase_key, json.dumps(event))
        
        # 아이템별 구매 수 업데이트
        purchase_count_key = f"item:{item_id}:purchases"
        self.redis_client.incr(purchase_count_key)
        
        # 전환율 계산을 위한 데이터 업데이트
        conversion_key = f"item:{item_id}:conversions"
        self.redis_client.incr(conversion_key)
    
    def update_user_features(self, user_id, event):
        """사용자 특성 실시간 업데이트"""
        user_features_key = f"user:{user_id}:features"
        
        # 현재 특성 조회
        current_features = self.redis_client.hgetall(user_features_key)
        current_features = {k.decode(): float(v.decode()) 
                          for k, v in current_features.items()}
        
        # 특성 업데이트 로직
        if event['event_type'] == 'click':
            current_features['total_clicks'] = current_features.get('total_clicks', 0) + 1
            current_features['last_click_timestamp'] = event['timestamp']
        elif event['event_type'] == 'purchase':
            current_features['total_purchases'] = current_features.get('total_purchases', 0) + 1
            current_features['total_spent'] = current_features.get('total_spent', 0) + event['amount']
        
        # Redis에 업데이트된 특성 저장
        self.redis_client.hmset(user_features_key, current_features)
        self.redis_client.expire(user_features_key, 86400 * 30)  # 30일 TTL
    
    def send_monitoring_data(self, event):
        """모니터링 시스템에 데이터 전송"""
        monitoring_event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event['event_type'],
            'user_id': event['user_id'],
            'item_id': event['item_id'],
            'metadata': event.get('metadata', {})
        }
        
        self.producer.send('monitoring-events', monitoring_event)

# 실시간 데이터 프로세서 실행
processor = RealtimeDataProcessor()
processor.process_user_events()
```

**3단계: 모델 훈련 파이프라인 (Apache Airflow)**
```python
# airflow_dag.py - 일일 모델 재학습 DAG
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import mlflow
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'recommendation_model_training',
    default_args=default_args,
    description='일일 추천 모델 재학습',
    schedule_interval='0 2 * * *',  # 매일 새벽 2시
    catchup=False
)

def extract_training_data(**context):
    """훈련 데이터 추출"""
    import cassandra
    from cassandra.cluster import Cluster
    
    # Cassandra에서 사용자 행동 데이터 추출
    cluster = Cluster(['127.0.0.1'])
    session = cluster.connect('recommendation')
    
    # 지난 30일간의 데이터 추출
    query = """
    SELECT user_id, item_id, event_type, timestamp, rating
    FROM user_events
    WHERE timestamp >= ?
    """
    
    thirty_days_ago = datetime.now() - timedelta(days=30)
    rows = session.execute(query, [thirty_days_ago])
    
    df = pd.DataFrame(rows)
    
    # 데이터 저장
    df.to_parquet('/tmp/training_data.parquet')
    return '/tmp/training_data.parquet'

def train_collaborative_filtering_model(**context):
    """협업 필터링 모델 훈련"""
    with mlflow.start_run():
        # 데이터 로드
        df = pd.read_parquet('/tmp/training_data.parquet')
        
        # 사용자-아이템 매트릭스 생성
        user_item_matrix = df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating',
            fill_value=0
        )
        
        # 협업 필터링 모델 (User-based)
        user_similarity = cosine_similarity(user_item_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity, 
            index=user_item_matrix.index, 
            columns=user_item_matrix.index
        )
        
        # 모델 평가
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        
        # 평가 데이터 분할
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        
        # 예측 성능 평가
        predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            # 유사한 사용자들의 평점으로 예측
            if user_id in user_similarity_df.index:
                similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]  # Top 10
                
                predicted_rating = 0
                weight_sum = 0
                
                for similar_user, similarity in similar_users.items():
                    if item_id in user_item_matrix.columns and similarity > 0:
                        user_rating = user_item_matrix.loc[similar_user, item_id]
                        if user_rating > 0:
                            predicted_rating += similarity * user_rating
                            weight_sum += similarity
                
                if weight_sum > 0:
                    predicted_rating /= weight_sum
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)
        
        # 성능 메트릭 계산
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        # MLflow에 메트릭 로깅
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mse", mse)
        mlflow.log_param("model_type", "collaborative_filtering")
        mlflow.log_param("similarity_metric", "cosine")
        
        # 모델 아티팩트 저장
        import joblib
        model_artifacts = {
            'user_item_matrix': user_item_matrix,
            'user_similarity': user_similarity_df
        }
        
        joblib.dump(model_artifacts, '/tmp/cf_model.pkl')
        mlflow.log_artifact('/tmp/cf_model.pkl')
        
        return rmse

def train_content_based_model(**context):
    """컨텐츠 기반 모델 훈련"""
    with mlflow.start_run():
        # 아이템 특성 데이터 로드
        item_features = pd.read_parquet('/tmp/item_features.parquet')
        
        # TF-IDF를 사용한 아이템 유사도 계산
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel
        
        # 아이템 설명으로부터 TF-IDF 벡터 생성
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(item_features['description'])
        
        # 코사인 유사도 계산
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        # 모델 성능 평가 (다양성 및 커버리지 측정)
        diversity_score = np.mean(1 - cosine_sim[np.triu_indices_from(cosine_sim, k=1)])
        coverage = len(np.unique(np.argmax(cosine_sim, axis=1))) / len(item_features)
        
        # MLflow에 메트릭 로깅
        mlflow.log_metric("diversity_score", diversity_score)
        mlflow.log_metric("coverage", coverage)
        mlflow.log_param("model_type", "content_based")
        mlflow.log_param("vectorizer", "tfidf")
        
        # 모델 아티팩트 저장
        import joblib
        model_artifacts = {
            'tfidf_vectorizer': tfidf,
            'cosine_similarity': cosine_sim,
            'item_features': item_features
        }
        
        joblib.dump(model_artifacts, '/tmp/content_model.pkl')
        mlflow.log_artifact('/tmp/content_model.pkl')
        
        return diversity_score

def model_validation(**context):
    """모델 검증 및 A/B 테스트 설정"""
    # 모델 성능 비교
    cf_rmse = context['task_instance'].xcom_pull(task_ids='train_cf_model')
    cb_diversity = context['task_instance'].xcom_pull(task_ids='train_content_model')
    
    # 성능 기준 검증
    if cf_rmse < 0.8 and cb_diversity > 0.3:  # 임계값
        print("✅ 모델 성능 기준 통과")
        
        # A/B 테스트 설정
        ab_config = {
            'experiment_name': f'recommendation_ab_{datetime.now().strftime("%Y%m%d")}',
            'models': {
                'model_a': 'collaborative_filtering',
                'model_b': 'content_based'
            },
            'traffic_split': {'model_a': 0.5, 'model_b': 0.5},
            'success_metric': 'ctr',
            'minimum_sample_size': 10000
        }
        
        # A/B 테스트 설정 저장
        with open('/tmp/ab_test_config.json', 'w') as f:
            json.dump(ab_config, f)
        
        return True
    else:
        print("❌ 모델 성능 기준 미달")
        return False

def deploy_models(**context):
    """모델 배포"""
    validation_passed = context['task_instance'].xcom_pull(task_ids='model_validation')
    
    if validation_passed:
        # TensorFlow Serving에 모델 배포
        import subprocess
        
        # 모델을 TensorFlow Serving 형식으로 변환 및 배포
        commands = [
            "docker stop recommendation-server-a recommendation-server-b",
            "docker rm recommendation-server-a recommendation-server-b",
            "docker run -d --name recommendation-server-a -p 8501:8501 -v /models/cf:/models/cf -e MODEL_NAME=cf tensorflow/serving",
            "docker run -d --name recommendation-server-b -p 8502:8501 -v /models/content:/models/content -e MODEL_NAME=content tensorflow/serving"
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"배포 명령 실패: {cmd}")
                raise
        
        print("✅ 모델 배포 완료")
    else:
        print("❌ 검증 실패로 배포 건너뛰기")

# Airflow 태스크 정의
extract_data_task = PythonOperator(
    task_id='extract_training_data',
    python_callable=extract_training_data,
    dag=dag
)

train_cf_task = PythonOperator(
    task_id='train_cf_model',
    python_callable=train_collaborative_filtering_model,
    dag=dag
)

train_content_task = PythonOperator(
    task_id='train_content_model',
    python_callable=train_content_based_model,
    dag=dag
)

validation_task = PythonOperator(
    task_id='model_validation',
    python_callable=model_validation,
    dag=dag
)

deploy_task = PythonOperator(
    task_id='deploy_models',
    python_callable=deploy_models,
    dag=dag
)

# 태스크 의존성 설정
extract_data_task >> [train_cf_task, train_content_task] >> validation_task >> deploy_task
```

**4단계: A/B 테스트 프레임워크**
```python
# ab_testing_framework.py
import random
import hashlib
import json
import redis
from datetime import datetime, timedelta
import pandas as pd

class ABTestingFramework:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=1)
        
    def assign_user_to_variant(self, user_id, experiment_name, variants):
        """사용자를 A/B 테스트 변형에 할당"""
        # 사용자 ID를 해싱하여 일관된 할당 보장
        user_hash = int(hashlib.md5(f"{user_id}_{experiment_name}".encode()).hexdigest(), 16)
        variant_index = user_hash % len(variants)
        
        assigned_variant = variants[variant_index]
        
        # Redis에 할당 정보 저장
        assignment_key = f"ab_test:{experiment_name}:{user_id}"
        self.redis_client.set(assignment_key, assigned_variant, ex=86400 * 30)  # 30일
        
        return assigned_variant
    
    def get_user_variant(self, user_id, experiment_name):
        """사용자의 할당된 변형 조회"""
        assignment_key = f"ab_test:{experiment_name}:{user_id}"
        variant = self.redis_client.get(assignment_key)
        return variant.decode() if variant else None
    
    def log_conversion_event(self, user_id, experiment_name, variant, event_type, value=1):
        """전환 이벤트 로깅"""
        event_data = {
            'user_id': user_id,
            'experiment_name': experiment_name,
            'variant': variant,
            'event_type': event_type,
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        
        # 이벤트 로그 저장
        event_key = f"ab_events:{experiment_name}"
        self.redis_client.lpush(event_key, json.dumps(event_data))
        
        # 집계 메트릭 업데이트
        metric_key = f"ab_metrics:{experiment_name}:{variant}:{event_type}"
        self.redis_client.incr(metric_key)
    
    def get_experiment_results(self, experiment_name):
        """실험 결과 조회"""
        event_key = f"ab_events:{experiment_name}"
        events = self.redis_client.lrange(event_key, 0, -1)
        
        if not events:
            return {'message': '실험 데이터 없음'}
        
        # 이벤트 데이터를 DataFrame으로 변환
        event_list = [json.loads(event.decode()) for event in events]
        df = pd.DataFrame(event_list)
        
        # 변형별 성능 분석
        results = {}
        
        for variant in df['variant'].unique():
            variant_data = df[df['variant'] == variant]
            
            # 기본 메트릭 계산
            total_users = variant_data['user_id'].nunique()
            total_impressions = len(variant_data[variant_data['event_type'] == 'impression'])
            total_clicks = len(variant_data[variant_data['event_type'] == 'click'])
            total_purchases = len(variant_data[variant_data['event_type'] == 'purchase'])
            
            # CTR 및 전환율 계산
            ctr = total_clicks / total_impressions if total_impressions > 0 else 0
            conversion_rate = total_purchases / total_users if total_users > 0 else 0
            
            results[variant] = {
                'users': total_users,
                'impressions': total_impressions,
                'clicks': total_clicks,
                'purchases': total_purchases,
                'ctr': ctr,
                'conversion_rate': conversion_rate
            }
        
        # 통계적 유의성 검정
        if len(results) == 2:
            from scipy.stats import chi2_contingency
            
            variants = list(results.keys())
            var_a, var_b = variants[0], variants[1]
            
            # CTR 비교 (카이제곱 검정)
            contingency_table = [
                [results[var_a]['clicks'], results[var_a]['impressions'] - results[var_a]['clicks']],
                [results[var_b]['clicks'], results[var_b]['impressions'] - results[var_b]['clicks']]
            ]
            
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            
            results['statistical_test'] = {
                'test': 'chi2_test',
                'chi2_statistic': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return results
    
    def stop_experiment(self, experiment_name, winning_variant=None):
        """실험 종료 및 승자 결정"""
        results = self.get_experiment_results(experiment_name)
        
        if winning_variant is None:
            # 자동으로 승자 결정 (CTR 기준)
            best_ctr = 0
            for variant, metrics in results.items():
                if isinstance(metrics, dict) and 'ctr' in metrics:
                    if metrics['ctr'] > best_ctr:
                        best_ctr = metrics['ctr']
                        winning_variant = variant
        
        # 실험 종료 정보 저장
        experiment_result = {
            'experiment_name': experiment_name,
            'end_time': datetime.now().isoformat(),
            'winning_variant': winning_variant,
            'final_results': results
        }
        
        result_key = f"ab_results:{experiment_name}"
        self.redis_client.set(result_key, json.dumps(experiment_result))
        
        print(f"실험 {experiment_name} 종료. 승자: {winning_variant}")
        return experiment_result

# A/B 테스트 프레임워크 사용 예시
ab_tester = ABTestingFramework()

# 추천 API에서 A/B 테스트 적용
def get_recommendations_with_ab_test(user_id, num_recommendations=10):
    experiment_name = "recommendation_model_comparison"
    variants = ['collaborative_filtering', 'content_based']
    
    # 사용자를 변형에 할당
    assigned_variant = ab_tester.assign_user_to_variant(user_id, experiment_name, variants)
    
    # 노출 이벤트 로깅
    ab_tester.log_conversion_event(user_id, experiment_name, assigned_variant, 'impression')
    
    # 할당된 변형에 따른 추천 생성
    if assigned_variant == 'collaborative_filtering':
        recommendations = get_cf_recommendations(user_id, num_recommendations)
    else:
        recommendations = get_content_recommendations(user_id, num_recommendations)
    
    return {
        'recommendations': recommendations,
        'variant': assigned_variant,
        'experiment': experiment_name
    }

def track_recommendation_click(user_id, item_id, experiment_name, variant):
    """추천 클릭 추적"""
    ab_tester.log_conversion_event(user_id, experiment_name, variant, 'click')

def track_recommendation_purchase(user_id, item_id, experiment_name, variant):
    """추천 구매 추적"""
    ab_tester.log_conversion_event(user_id, experiment_name, variant, 'purchase')
```

**5단계: 모니터링 및 알럿 시스템**
```python
# monitoring_system.py
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import time
import pandas as pd
from datetime import datetime, timedelta

# Prometheus 메트릭 정의
PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total predictions made', ['model_version', 'variant'])
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'Prediction latency')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy', ['model_version'])
CTR_GAUGE = Gauge('recommendation_ctr', 'Click-through rate', ['variant'])
DRIFT_SCORE = Gauge('data_drift_score', 'Data drift score', ['feature'])

class RecommendationMonitor:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=2)
        
    def track_prediction(self, model_version, variant, latency):
        """예측 추적"""
        PREDICTION_COUNTER.labels(model_version=model_version, variant=variant).inc()
        PREDICTION_LATENCY.observe(latency)
    
    def update_model_accuracy(self, model_version, accuracy):
        """모델 정확도 업데이트"""
        MODEL_ACCURACY.labels(model_version=model_version).set(accuracy)
    
    def update_ctr(self, variant, ctr):
        """CTR 업데이트"""
        CTR_GAUGE.labels(variant=variant).set(ctr)
    
    def update_drift_score(self, feature, drift_score):
        """드리프트 점수 업데이트"""
        DRIFT_SCORE.labels(feature=feature).set(drift_score)
    
    def check_performance_degradation(self):
        """성능 저하 체크"""
        # 최근 1시간 CTR 조회
        current_time = datetime.now()
        one_hour_ago = current_time - timedelta(hours=1)
        
        # Redis에서 최근 CTR 데이터 조회
        ctr_key = "hourly_ctr"
        recent_ctr_data = self.redis_client.zrangebyscore(
            ctr_key, 
            one_hour_ago.timestamp(), 
            current_time.timestamp(),
            withscores=True
        )
        
        if recent_ctr_data:
            recent_ctrs = [float(score) for _, score in recent_ctr_data]
            avg_ctr = sum(recent_ctrs) / len(recent_ctrs)
            
            # 임계값 비교 (예: 기준 CTR의 80% 이하)
            baseline_ctr = 0.05  # 5%
            threshold = baseline_ctr * 0.8
            
            if avg_ctr < threshold:
                self.send_alert(
                    "CTR 성능 저하",
                    f"최근 1시간 평균 CTR: {avg_ctr:.4f}, 임계값: {threshold:.4f}"
                )
                return True
        
        return False
    
    def check_latency_issues(self):
        """레이턴시 이슈 체크"""
        # Prometheus에서 최근 레이턴시 데이터 조회
        # (실제로는 Prometheus 클라이언트를 사용해야 함)
        recent_latencies = self.get_recent_latencies()  # 구현 필요
        
        if recent_latencies:
            p95_latency = np.percentile(recent_latencies, 95)
            
            # 95퍼센타일 레이턴시가 100ms 초과
            if p95_latency > 0.1:
                self.send_alert(
                    "레이턴시 임계값 초과",
                    f"95퍼센타일 레이턴시: {p95_latency*1000:.2f}ms"
                )
                return True
        
        return False
    
    def send_alert(self, title, message):
        """알럿 전송"""
        alert_data = {
            'title': title,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'severity': 'warning'
        }
        
        # Slack, Email, PagerDuty 등으로 알럿 전송
        print(f"🚨 ALERT: {title} - {message}")
        
        # 알럿 히스토리 저장
        alert_key = f"alerts:{datetime.now().strftime('%Y%m%d')}"
        self.redis_client.lpush(alert_key, json.dumps(alert_data))
    
    def run_monitoring_loop(self):
        """모니터링 루프 실행"""
        while True:
            try:
                # 성능 저하 체크
                if self.check_performance_degradation():
                    print("성능 저하 감지됨")
                
                # 레이턴시 이슈 체크
                if self.check_latency_issues():
                    print("레이턴시 이슈 감지됨")
                
                # 5분마다 체크
                time.sleep(300)
                
            except Exception as e:
                print(f"모니터링 에러: {e}")
                time.sleep(60)

# 모니터링 시스템 실행
monitor = RecommendationMonitor()
# monitor.run_monitoring_loop()  # 백그라운드에서 실행
```

**해설:**
이 예제는 실시간 추천 시스템을 위한 **종단간 MLOps 파이프라인**을 구현합니다:

1. **실시간 데이터 처리**: Kafka와 Redis를 활용한 실시간 사용자 행동 데이터 처리
2. **자동화된 모델 훈련**: Airflow를 통한 일일 배치 재학습
3. **A/B 테스트**: 두 모델 간의 성능을 실제 사용자 트래픽으로 비교
4. **실시간 모니터링**: Prometheus를 통한 메트릭 수집 및 알럿
5. **성능 보장**: 100ms 미만 레이턴시 요구사항 충족

---

## 핵심 요약 (Key Takeaways)

### MLOps의 핵심 가치
1. **자동화**: 반복적인 ML 작업의 자동화를 통한 효율성 증대
2. **신뢰성**: 체계적인 테스트와 검증을 통한 모델 품질 보장  
3. **확장성**: 대규모 서비스에서도 안정적인 모델 서빙
4. **협업**: 다양한 역할 간의 효과적인 협업 체계 구축

### 성공적인 MLOps 구축 요소
1. **문화**: DevOps 문화의 ML팀 확산
2. **프로세스**: 표준화된 워크플로우와 품질 관리
3. **도구**: 적절한 MLOps 도구 스택 선택과 통합
4. **측정**: 비즈니스 가치와 연결된 메트릭 정의

### 단계별 MLOps 성숙도
- **Level 0**: 수동 프로세스 (실험 단계)
- **Level 1**: ML 파이프라인 자동화 (CT 도입)
- **Level 2**: CI/CD 파이프라인 자동화 (완전 자동화)

MLOps는 머신러닝 모델을 **연구실에서 실제 비즈니스 가치로** 연결하는 핵심 방법론입니다. 성공적인 MLOps 구축을 통해 조직은 더 빠르고 안정적으로 AI의 가치를 실현할 수 있습니다.
