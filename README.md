# AI/ML 및 로봇공학 학습을 위한 종합 커리큘럼

## 파트 1: AI/ML 학습을 위한 기초 (Foundations)

### 1.1. 수학 (Mathematics)


#### 1.1.1. 선형대수 (Linear Algebra)
- [ ] [스칼라, 벡터, 행렬, 텐서의 이해](./docs/01_01_01_Scalar_Vector_Matrix_Tensor.md)
- [ ] [행렬 연산 (덧셈, 뺄셈, 곱셈, 전치)](./docs/01_01_02_Matrix_Operations.md)
- [ ] [내적(Dot Product) 및 외적(Cross Product)](./docs/01_01_03_Dot_Product_and_Cross_Product.md)
- [ ] [단위행렬(Identity Matrix)과 역행렬(Inverse Matrix)](./docs/01_01_04_Identity_and_Inverse_Matrix.md)
- [ ] [행렬식(Determinant)](./docs/01_01_05_Determinant.md)
- [ ] [고유값(Eigenvalues)과 고유벡터(Eigenvectors)](./docs/01_01_06_Eigenvalues_and_Eigenvectors.md)
- [ ] [특이값 분해 (Singular Value Decomposition, SVD)](./docs/01_01_07_Singular_Value_Decomposition.md)
- [ ] [주성분 분석 (Principal Component Analysis, PCA) - 기초 개념](./docs/01_01_08_Principal_Component_Analysis.md)


#### 1.1.2. 미적분 (Calculus)
- [ ] [극한(Limits)과 연속(Continuity)](./docs/01_02_01_Limits_and_Continuity.md)
- [ ] [미분(Derivatives)과 미분계수](./docs/01_02_02_Derivatives.md)
- [ ] [편미분(Partial Derivatives)](./docs/01_02_03_Partial_Derivatives.md)
- [ ] [연쇄 법칙 (Chain Rule)](./docs/01_02_04_Chain_Rule.md)
- [ ] [그래디언트 (Gradient)](./docs/01_02_05_Gradient.md)
- [ ] [적분(Integrals)](./docs/01_02_06_Integrals.md)
- [ ] [시그모이드(Sigmoid), ReLU 등 활성화 함수의 미분](./docs/01_02_07_Activation_Function_Derivatives.md)

#### 1.1.3. 확률 및 통계 (Probability & Statistics)
- [ ] 기본적인 확률 이론 (사건, 표본 공간, 조건부 확률)
- [ ] 베이즈 정리 (Bayes' Theorem)
- [ ] 확률 변수 (Random Variables)
- [ ] 확률 분포 (균등, 정규, 베르누이, 이항, 포아송 분포)
- [ ] 기댓값(Expected Value), 분산(Variance), 표준편차(Standard Deviation)
- [ ] 중심극한정리 (Central Limit Theorem)
- [ ] 기술 통계 (Descriptive Statistics): 평균, 중앙값, 최빈값
- [ ] 추론 통계 (Inferential Statistics): 가설 검정, 신뢰 구간
- [ ] 최대 가능도 추정 (Maximum Likelihood Estimation, MLE)

#### 1.1.4. 최적화 (Optimization)
- [ ] 비용 함수 (Cost Function) / 손실 함수 (Loss Function)
- [ ] 경사 하강법 (Gradient Descent) 및 변형 (Stochastic, Mini-batch)
- [ ] 볼록 최적화 (Convex Optimization)
- [ ] 라그랑주 승수법 (Lagrange Multipliers)

### 1.2. 컴퓨터 과학 (Computer Science)

#### 1.2.1. 프로그래밍 기초 (Programming Fundamentals with Python)
- [ ] 파이썬 기본 문법 (변수, 자료형, 연산자)
- [ ] 제어문 (조건문, 반복문)
- [ ] 함수와 모듈
- [ ] 객체 지향 프로그래밍 (OOP) 기초 (클래스, 객체)
- [ ] 파일 입출력
- [ ] 필수 라이브러리:
  - [ ] **NumPy**: 다차원 배열 및 행렬 연산
  - [ ] **Pandas**: 데이터 조작 및 분석 (DataFrame, Series)
  - [ ] **Matplotlib/Seaborn**: 데이터 시각화

#### 1.2.2. 자료구조 및 알고리즘 (Data Structures & Algorithms)
- [ ] 시간 복잡도와 공간 복잡도 (Big O 표기법)
- [ ] 기본 자료구조: 배열, 연결 리스트, 스택, 큐
- [ ] 고급 자료구조: 트리, 힙, 해시 테이블, 그래프
- [ ] 정렬 알고리즘 (버블, 선택, 삽입, 병합, 퀵 정렬)
- [ ] 탐색 알고리즘 (선형, 이진 탐색)
- [ ] 그래프 순회 (너비 우선 탐색 - BFS, 깊이 우선 탐색 - DFS)

## 파트 2: 인공지능(AI) 및 머신러닝(ML) 소개 (Introduction to AI & ML)

### 2.1. 인공지능의 역사와 발전 (History and Evolution of AI)
- [x] [튜링 테스트와 AI의 탄생 (앨런 튜링)](./docs/02_01_01_Turing_Test_and_Birth_of_AI.md)
- [x] [다트머스 회의: '인공지능' 용어의 등장](./docs/02_01_02_Dartmouth_Conference.md)
- [x] [AI의 황금기와 암흑기 (AI Winters)](./docs/02_01_03_AI_Golden_Age_and_Winter.md)
- [x] [전문가 시스템(Expert Systems)의 시대](./docs/02_01_04_Expert_Systems.md)
- [x] [머신러닝의 부상과 딥러닝 혁명](./docs/02_01_05_Rise_of_ML_and_Deep_Learning.md)

### 2.2. AI, 머신러닝, 딥러닝의 관계 (Relationship between AI, ML, and Deep Learning)
- [x] [AI, 머신러닝, 딥러닝의 관계](./docs/02_02_01_Relationship_AI_ML_DL.md)

### 2.3. 머신러닝의 종류 (Types of Machine Learning)
- [x] [지도학습 (Supervised Learning)](./docs/02_03_01_Supervised_Learning.md)
- [x] [비지도학습 (Unsupervised Learning)](./docs/02_03_02_Unsupervised_Learning.md)
- [x] [강화학습 (Reinforcement Learning)](./docs/02_03_03_Reinforcement_Learning.md)

### 2.4. 머신러닝 프로젝트의 전체 과정 (The Machine Learning Workflow)
- [x] [머신러닝 프로젝트의 전체 과정](./docs/02_04_01_ML_Workflow.md)

## 파트 3: 핵심 머신러닝 (Core Machine Learning)

### 3.1. 지도 학습 (Supervised Learning)

#### 3.1.1. 회귀 (Regression)
- [x] [**선형 회귀 (Linear Regression):** 기본 개념, 비용 함수, 경사 하강법 적용](./docs/03_01_01_Linear_Regression.md)
- [x] [**다항 회귀 (Polynomial Regression):** 비선형 관계 모델링](./docs/03_01_01_Polynomial_Regression.md)
- [x] [**규제가 있는 회귀 (Regularized Regression):**](./docs/03_01_01_Regularized_Regression.md)
  - [ ] **릿지 회귀 (Ridge Regression):** L2 규제
  - [ ] **라쏘 회귀 (Lasso Regression):** L1 규제
  - [ ] **엘라스틱넷 (ElasticNet):** L1과 L2 규제의 결합

#### 3.1.2. 분류 (Classification)
- [x] [**로지스틱 회귀 (Logistic Regression):** 시그모이드 함수, 이진 분류](./docs/03_01_02_Logistic_Regression.md)
- [x] [**k-최근접 이웃 (k-Nearest Neighbors, k-NN):** 거리 기반 모델, 인스턴스 기반 학습](./docs/03_01_02_kNN.md)
- [x] [**서포트 벡터 머신 (Support Vector Machines, SVM):** 마진 최대화, 커널 트릭](./docs/03_01_02_SVM.md)
- [x] [**결정 트리 (Decision Trees):** 정보 이득, 지니 불순도, 가지치기](./docs/03_01_02_Decision_Trees.md)
- [x] [**나이브 베이즈 분류기 (Naive Bayes Classifier):** 베이즈 정리 기반, 특성 독립 가정](./docs/03_01_02_Naive_Bayes.md)

#### 3.1.3. 앙상블 학습 (Ensemble Learning)
- [x] [**보팅 (Voting) 및 배깅 (Bagging):**](./docs/03_01_03_Voting_and_Bagging.md)
  - [ ] **보팅:** 여러 모델의 예측을 투표로 결합
  - [ ] **배깅:** 데이터 샘플링을 통해 여러 모델을 학습 (예: 부트스트랩)
- [x] [**랜덤 포레스트 (Random Forest):** 결정 트리의 앙상블 (배깅 + 특성 무작위 선택)](./docs/03_01_03_Random_Forest.md)
- [x] [**부스팅 (Boosting):**](./docs/03_01_03_Boosting.md)
  - [ ] **AdaBoost:** 오답에 가중치를 부여하며 순차적으로 학습
  - [ ] **그래디언트 부스팅 머신 (Gradient Boosting Machine, GBM):** 잔여 오차를 학습
  - [ ] **XGBoost, LightGBM, CatBoost:** GBM의 성능 및 속도 개선 모델

### 3.2. 비지도 학습 (Unsupervised Learning)

#### 3.2.1. 군집화 (Clustering)
- [ ] **K-평균 군집화 (K-Means Clustering):** 중심점(Centroid) 기반 군집화
- [ ] **계층적 군집화 (Hierarchical Clustering):** 덴드로그램, 병합/분할 방식
- [ ] **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** 밀도 기반 군집화

#### 3.2.2. 차원 축소 (Dimensionality Reduction)
- [ ] **주성분 분석 (Principal Component Analysis, PCA):** 분산을 최대로 보존하는 축을 찾아 차원 축소
- [ ] **t-SNE (t-Distributed Stochastic Neighbor Embedding):** 고차원 데이터의 시각화에 주로 사용

### 3.3. 모델 평가 및 성능 향상 (Model Evaluation & Enhancement)

#### 3.3.1. 핵심 개념 (Key Concepts)
- [ ] **과대적합 (Overfitting) 과 과소적합 (Underfitting):** 모델의 일반화 능력
- [ ] **편향-분산 트레이드오프 (Bias-Variance Tradeoff):** 모델 복잡도와의 관계
- [ ] **교차 검증 (Cross-Validation):** K-Fold, Stratified K-Fold

#### 3.3.2. 회귀 모델 평가 지표 (Regression Metrics)
- [ ] **MAE (Mean Absolute Error):** 평균 절대 오차
- [ ] **MSE (Mean Squared Error):** 평균 제곱 오차
- [ ] **RMSE (Root Mean Squared Error):** 평균 제곱근 오차
- [ ] **R² (R-squared):** 결정 계수 (모델의 설명력)

#### 3.3.3. 분류 모델 평가 지표 (Classification Metrics)
- [ ] **혼동 행렬 (Confusion Matrix):** TP, FP, FN, TN
- [ ] **정확도 (Accuracy):** 전체 예측 중 올바른 예측의 비율
- [ ] **정밀도 (Precision):** Positive로 예측한 것 중 실제 Positive의 비율
- [ ] **재현율 (Recall) / 민감도 (Sensitivity):** 실제 Positive 중 Positive로 예측한 비율
- [ ] **F1 점수 (F1-Score):** 정밀도와 재현율의 조화 평균
- [ ] **ROC 곡선과 AUC (Area Under the Curve):** 분류 모델의 성능을 종합적으로 평가

## 파트 4: 딥러닝 (Deep Learning)

### 4.1. 인공 신경망 기초 (Neural Network Basics)
- [x] [**퍼셉트론 (Perceptron)과 다층 퍼셉트론 (Multi-Layer Perceptron, MLP):** 신경망의 기본 단위](./docs/04_01_01_Perceptron_and_MLP.md)
- [x] [**활성화 함수 (Activation Functions):**](./docs/04_01_02_Activation_Functions.md)
  - Sigmoid, Tanh, ReLU, Leaky ReLU, PReLU, ELU, Softmax
- [x] [**손실 함수 (Loss Functions):**](./docs/04_01_03_Loss_Functions.md)
  - 회귀: MSE, MAE
  - 분류: Binary Cross-Entropy, Categorical Cross-Entropy
- [x] [**역전파 알고리즘 (Backpropagation):** 그래디언트 계산 및 가중치 업데이트 원리](./docs/04_01_04_Backpropagation.md)
- [x] [**옵티마이저 (Optimizers):**](./docs/04_01_05_Optimizers.md)
  - 경사 하강법의 종류: SGD, Momentum, Nesterov Momentum
  - 적응적 학습률 옵티마이저: Adagrad, RMSprop, Adam

### 4.2. 딥러닝 모델 학습 기술 (Techniques for Training Deep Neural Networks)
- [x] [**가중치 초기화 (Weight Initialization):** Xavier/Glorot, He 초기화](./docs/04_02_01_Weight_Initialization.md)
- [x] [**배치 정규화 (Batch Normalization):** 내부 공변량 변화 문제 해결](./docs/04_02_02_Batch_Normalization.md)
- [x] [**드롭아웃 (Dropout):** 과대적합 방지를 위한 규제 기법](./docs/04_02_03_Dropout.md)
- [x] [**학습률 스케줄링 (Learning Rate Scheduling):** Step Decay, Cosine Annealing 등](./docs/04_02_04_Learning_Rate_Scheduling.md)

### 4.3. 합성곱 신경망 (Convolutional Neural Networks, CNN)
- [x] [**CNN의 핵심 구성요소:**](./docs/04_03_01_CNN_Components.md)
  - **합성곱 (Convolution) 레이어:** 필터(Filter)/커널(Kernel), 스트라이드(Stride), 패딩(Padding)
  - **풀링 (Pooling) 레이어:** Max Pooling, Average Pooling
- [x] [**주요 CNN 아키텍처:**](./docs/04_03_02_CNN_Architectures.md)
  - **LeNet-5, AlexNet, VGGNet, GoogLeNet (Inception), ResNet (Residual Networks), DenseNet**
- [x] [**응용 분야:**](./docs/04_03_03_CNN_Applications.md)
  - **이미지 분류 (Image Classification)**
  - **객체 탐지 (Object Detection):** R-CNN, Fast R-CNN, Faster R-CNN, YOLO, SSD
  - **이미지 분할 (Image Segmentation):** FCN, U-Net

### 4.4. 순환 신경망 (Recurrent Neural Networks, RNN)
- [x] [**순차 데이터 (Sequential Data)의 이해:** 시계열, 텍스트 데이터](./docs/04_04_01_Sequential_Data.md)
- [x] [**RNN의 기본 구조와 한계:** 기울기 소실/폭주 문제 (Vanishing/Exploding Gradient Problem)](./docs/04_04_02_RNN_and_Limits.md)
- [x] [**LSTM (Long Short-Term Memory):** Cell State, Forget/Input/Output Gate를 통한 장기 의존성 학습](./docs/04_04_03_LSTM.md)
- [x] [**GRU (Gated Recurrent Unit):** LSTM을 단순화한 모델](./docs/04_04_04_GRU.md)
- [x] [**양방향 RNN (Bidirectional RNN):** 과거와 미래 정보를 모두 활용](./docs/04_04_05_Bidirectional_RNN.md)
- [x] [**응용 분야:** 자연어 처리(NLP), 시계열 예측](./docs/04_04_06_RNN_Applications.md)

### 4.5. 트랜스포머와 어텐션 메커니즘 (Transformers and Attention Mechanism)
- [x] [**어텐션 메커니즘 (Attention Mechanism):** Seq2Seq 모델의 한계 극복](./docs/04_05_01_Attention_Mechanism.md)
- [x] [**트랜스포머 (Transformer) 아키텍처:**](./docs/04_05_02_Transformer_Architecture.md)
  - **Self-Attention, Multi-Head Attention**
  - **Positional Encoding**
  - **인코더-디코더 구조**
- [x] [**주요 트랜스포머 기반 모델:**](./docs/04_05_03_Transformer_Models.md)
  - **BERT (Bidirectional Encoder Representations from Transformers)**
  - **GPT (Generative Pre-trained Transformer)**
  - **T5, BART**
- [x] [**응용 분야:** 기계 번역, 텍스트 생성, 요약, 질의응답](./docs/04_05_04_Transformer_Applications.md)

### 4.6. 생성 모델 (Generative Models)
- [x] [**오토인코더 (Autoencoders, AE):** 데이터 압축 및 특징 추출](./docs/04_06_01_Autoencoders.md)
- [x] [**변이형 오토인코더 (Variational Autoencoders, VAE):** 확률적 잠재 공간을 이용한 데이터 생성](./docs/04_06_02_Variational_Autoencoders.md)
- [x] [**생성적 적대 신경망 (Generative Adversarial Networks, GAN):**](./docs/04_06_03_GANs.md)
  - 생성자(Generator)와 판별자(Discriminator)의 경쟁을 통한 학습
  - DCGAN, StyleGAN 등 주요 GAN 모델
- [x] [**디퓨전 모델 (Diffusion Models):** 노이즈를 점진적으로 제거하며 데이터 생성](./docs/04_06_04_Diffusion_Models.md)
- [x] [**응용 분야:** 이미지 생성, 스타일 변환(Style Transfer), 데이터 증강](./docs/04_06_05_Generative_Applications.md)

### 4.7. 그래프 신경망 (Graph Neural Networks, GNN)
- [x] [**그래프 데이터의 이해:** 노드, 엣지, 그래프 구조](./docs/04_07_01_Graph_Data.md)
- [x] [**GNN의 기본 원리:** 메시지 전달(Message Passing)](./docs/04_07_02_GNN_Principles.md)
- [x] [**주요 GNN 아키텍처:** GCN, GraphSAGE, GAT](./docs/04_07_03_GNN_Architectures.md)
- [x] [**응용 분야:** 추천 시스템, 소셜 네트워크 분석, 분자 구조 예측](./docs/04_07_04_GNN_Applications.md)

## 파트 5: 강화학습 (Reinforcement Learning)

### 5.1. 강화학습 기초 (Fundamentals of Reinforcement Learning)
- [ ] [**주요 구성요소:** 에이전트(Agent), 환경(Environment), 상태(State), 행동(Action), 보상(Reward)](./docs/05_01_01_Key_Components.md)
- [ ] [**정책 (Policy):** 상태에 따라 에이전트가 행동을 결정하는 방식 (π)](./docs/05_01_02_Policy.md)
- [ ] [**가치 함수 (Value Function):** 특정 상태 또는 상태-행동 쌍의 가치를 평가 (V-function, Q-function)](./docs/05_01_03_Value_Function.md)
- [ ] [**모델 (Model):** 환경이 어떻게 동작할지 예측 (상태 전이, 보상)](./docs/05_01_04_Model.md)
- [ ] [**탐험과 활용의 딜레마 (Exploration vs. Exploitation Tradeoff)**](./docs/05_01_05_Exploration_vs_Exploitation.md)

### 5.2. 마르코프 결정 과정 (Markov Decision Processes, MDP)
- [ ] [**마르코프 속성 (Markov Property):** 현재 상태가 과거의 모든 정보를 포함](./docs/05_02_01_Markov_Decision_Processes.md)
- [ ] [**구성요소:** 상태(S), 행동(A), 상태 전이 확률(P), 보상 함수(R), 감가율(γ)](./docs/05_02_01_Markov_Decision_Processes.md)
- [ ] [**벨만 방정식 (Bellman Equations):**](./docs/05_02_02_Bellman_Equations.md)
  - [ ] [**벨만 기대 방정식 (Bellman Expectation Equation):** 현재 가치와 다음 상태 가치의 관계](./docs/05_02_02_Bellman_Equations.md)
  - [ ] [**벨만 최적 방정식 (Bellman Optimality Equation):** 최적 가치 함수를 찾기 위한 방정식](./docs/05_02_02_Bellman_Equations.md)

### 5.3. 주요 강화학습 알고리즘 (Key Reinforcement Learning Algorithms)

#### 5.3.1. 모델 프리 학습 (Model-Free Learning)
- [ ] [환경의 모델(상태 전이, 보상)을 모르는 상태에서 학습](./docs/05_03_01_Model_Free_Learning.md)

#### 5.3.2. 가치 기반 학습 (Value-Based Learning)
- [ ] [최적 가치 함수를 학습하여 최적 정책을 유도](./docs/05_03_02_Value_Based_Learning.md)
- [ ] [**Q-러닝 (Q-Learning):** 오프-폴리시(Off-policy) 학습](./docs/05_03_02_Value_Based_Learning.md)
- [ ] [**SARSA (State-Action-Reward-State-Action):** 온-폴리시(On-policy) 학습](./docs/05_03_02_Value_Based_Learning.md)
- [ ] [**심층 Q-네트워크 (Deep Q-Network, DQN):** 신경망을 이용해 Q-함수를 근사](./docs/05_03_02_Value_Based_Learning.md)
  - [ ] [**Experience Replay, Target Network**](./docs/05_03_02_Value_Based_Learning.md)

#### 5.3.3. 정책 기반 학습 (Policy-Based Learning)
- [ ] [최적 정책을 직접 학습](./docs/05_03_03_Policy_Based_Learning.md)
- [ ] [**정책 경사 (Policy Gradient, PG):** 정책의 성능을 나타내는 목적 함수를 경사 상승법으로 최적화](./docs/05_03_03_Policy_Based_Learning.md)
- [ ] [**REINFORCE 알고리즘**](./docs/05_03_03_Policy_Based_Learning.md)

#### 5.3.4. 액터-크리틱 (Actor-Critic Methods)
- [ ] [가치 기반과 정책 기반 학습의 결합](./docs/05_03_04_Actor_Critic_Methods.md)
- [ ] [**액터 (Actor):** 정책을 학습 (어떻게 행동할지 결정)](./docs/05_03_04_Actor_Critic_Methods.md)
- [ ] [**크리틱 (Critic):** 가치 함수를 학습 (액터가 한 행동을 평가)](./docs/05_03_04_Actor_Critic_Methods.md)
- [ ] [**A2C (Advantage Actor-Critic) / A3C (Asynchronous Advantage Actor-Critic)**](./docs/05_03_04_Actor_Critic_Methods.md)
- [ ] [**DDPG (Deep Deterministic Policy Gradient):** 연속적인 행동 공간에서 사용](./docs/05_03_04_Actor_Critic_Methods.md)
- [ ] [**SAC (Soft Actor-Critic)**](./docs/05_03_04_Actor_Critic_Methods.md)

## 파트 6: MLOps (Machine Learning Operations)

### 6.1. MLOps 개요 (Introduction to MLOps)
- [x] [MLOps의 정의, 필요성, DevOps와의 차이점, 전체 파이프라인](./docs/06_01_Introduction_to_MLOps.md)

### 6.2. 데이터 및 피처 관리 (Data and Feature Management)
- [x] [데이터 파이프라인, 데이터 버전 관리(DVC), 피처 스토어](./docs/06_02_Data_and_Feature_Management.md)

### 6.3. 모델 배포 및 서빙 (Model Deployment & Serving)
- [x] [배포 전략, 모델 서빙, 컨테이너화, 서버리스](./docs/06_03_Model_Deployment_and_Serving.md)

### 6.4. CI/CD/CT for ML
- [x] [지속적 통합(CI), 지속적 제공/배포(CD), 지속적 학습(CT)](./docs/06_04_CI_CD_CT_for_ML.md)

### 6.5. 모델 모니터링 및 관리 (Model Monitoring & Management)
- [x] [모델 성능 모니터링, 드리프트, 모델 레지스트리, MLOps 플랫폼](./docs/06_05_Model_Monitoring_and_Management.md)

## 파트 7: 심화 주제 및 전문 분야 (Advanced Topics & Specializations)

### 7.1. 자연어 처리 심화 (Advanced Natural Language Processing, NLP)
- [x] [**텍스트 임베딩 (Text Embeddings)**](./docs/07_01_01_Text_Embeddings.md)
  - [x] Word2Vec, GloVe, FastText: 단어 수준의 벡터 표현
  - [x] Contextual Embeddings: ELMo, BERT, GPT
- [x] [**주요 NLP 과제**](./docs/07_01_02_NLP_Tasks.md)
  - [x] 감성 분석 (Sentiment Analysis)
  - [x] 개체명 인식 (Named Entity Recognition, NER)
  - [x] 토픽 모델링 (Topic Modeling): Latent Dirichlet Allocation (LDA)
  - [x] 기계 번역 (Machine Translation)
  - [x] 텍스트 요약 (Text Summarization)
  - [x] 질의응답 시스템 (Question Answering Systems)

### 7.2. 컴퓨터 비전 심화 (Advanced Computer Vision, CV)
- [x] [**전이 학습 (Transfer Learning) 및 미세 조정 (Fine-tuning)**](./docs/07_02_01_Transfer_Learning_and_Fine_tuning.md): 사전 훈련된 모델 활용
- [x] [**이미지 생성 (Image Generation) 및 스타일 변환 (Style Transfer)**](./docs/07_02_02_Image_Generation_and_Style_Transfer.md)
- [x] [**Semantic/Instance Segmentation 심화**](./docs/07_02_03_Advanced_Segmentation.md)
- [x] [**3D 비전 (3D Vision)**](./docs/07_02_04_3D_Vision.md)
- [x] [**비디오 분석 (Video Analysis)**](./docs/07_02_05_Video_Analysis.md)

### 7.3. 추천 시스템 (Recommender Systems)
- [x] [**콘텐츠 기반 필터링 (Content-Based Filtering)**](./docs/07_03_01_Content_Based_Filtering.md)
- [x] [**협업 필터링 (Collaborative Filtering)**](./docs/07_03_02_Collaborative_Filtering.md)
  - [x] 최근접 이웃 기반 (Neighborhood-based)
  - [x] 잠재 요인 모델 (Latent Factor Models): 행렬 분해 (Matrix Factorization), SVD
- [x] [**딥러닝 기반 추천 시스템 (Deep Learning-based Recommender Systems)**](./docs/07_03_03_Deep_Learning_based_Recommender_Systems.md)
- [x] [**하이브리드 모델 (Hybrid Models)**](./docs/07_03_04_Hybrid_Models.md)

### 7.4. 시계열 분석 (Time Series Analysis)
- [x] [**전통적 시계열 모델**](./docs/07_04_01_Traditional_Time_Series_Models.md)
  - [x] AR (Autoregressive), MA (Moving Average), ARMA, ARIMA, SARIMA
- [x] [**딥러닝 기반 시계열 모델**](./docs/07_04_02_Deep_Learning_based_Time_Series_Models.md)
  - [x] RNNs, LSTMs, GRUs for Time Series Forecasting
- [x] [**기타 모델**](./docs/07_04_03_Other_Time_Series_Models.md): Facebook Prophet, N-BEATS

### 7.5. 책임감 있고 윤리적인 AI (Responsible & Ethical AI)
- [x] [**AI의 편향성과 공정성 (Bias and Fairness in AI)**](./docs/07_05_01_Bias_and_Fairness_in_AI.md)
  - [x] 편향의 종류와 탐지 방법
  - [x] 공정성 지표 및 완화 기법
- [x] [**설명 가능한 AI (Explainable AI, XAI)**](./docs/07_05_02_Explainable_AI_XAI.md)
  - [x] 모델의 예측을 인간이 이해할 수 있도록 설명
  - [x] LIME (Local Interpretable Model-agnostic Explanations)
  - [x] SHAP (SHapley Additive exPlanations)
- [x] [**프라이버시 보호 (Privacy in AI)**](./docs/07_05_03_Privacy_in_AI.md)
  - [x] 연합 학습 (Federated Learning)
  - [x] 차등 정보보호 (Differential Privacy)
- [x] [**AI 안전성 (AI Safety)**](./docs/07_05_04_AI_Safety.md): AI 시스템의 의도치 않은 행동 방지

## 파트 8: 로봇공학 기초 (Robotics Foundations)

### 8.1. 로봇 시스템 소개 (Introduction to Robot Systems)
- [x] [로봇의 정의, 종류 및 구성요소](./docs/08_01_01_Robot_Definition_Types_Components.md) (매니퓰레이터, 모바일 로봇, 휴머노이드)
- [x] [좌표계 (Coordinate Systems) 및 변환 (Transformations)](./docs/08_01_02_Coordinate_Systems_Transformations.md): 3D 회전 및 변환 행렬
- [x] [로봇의 자유도 (Degrees of Freedom, DoF)](./docs/08_01_03_Degrees_of_Freedom.md)

### 8.2. 로봇 운동학 (Robot Kinematics)
- [x] [순기구학 (Forward Kinematics)](./docs/08_02_01_Forward_Kinematics.md): Denavit-Hartenberg (DH) 파라미터
- [x] [역기구학 (Inverse Kinematics)](./docs/08_02_02_Inverse_Kinematics.md): 해석적 해법 (Analytical Solutions), 수치적 해법 (Numerical Solutions)
- [x] [자코비안 (Jacobian)](./docs/08_02_03_Jacobian.md): 속도 및 정적 힘(Static Force) 관계 분석

### 8.3. 로봇 동역학 (Robot Dynamics)
- [x] [라그랑주 역학 (Lagrangian Dynamics) 기반 모델링](./docs/08_03_01_Lagrangian_Dynamics.md)
- [x] [뉴턴-오일러 방정식 (Newton-Euler Formulation) 기반 모델링](./docs/08_03_02_Newton_Euler_Formulation.md)

### 8.4. 로봇 제어 (Robot Control)
- [x] [선형 제어 시스템 기초](./docs/08_04_01_Linear_Control_Systems.md): 전달 함수, 상태 공간 모델
- [x] [PID 제어 (Proportional-Integral-Derivative Control) 및 튜닝](./docs/08_04_02_PID_Control.md)
- [x] [궤적 추종 제어 (Trajectory Tracking Control)](./docs/08_04_03_Trajectory_Tracking_Control.md)
- [x] [힘 제어 (Force Control) 및 임피던스 제어 (Impedance Control)](./docs/08_04_04_Force_Impedance_Control.md)

### 8.5. 로봇 센서 및 액추에이터 (Sensors and Actuators)
- [x] [주요 센서](./docs/08_05_01_Robot_Sensors.md): 카메라, LiDAR, RADAR, IMU, GPS, 엔코더(Encoder), 힘/토크 센서
- [x] [주요 액추에이터](./docs/08_05_02_Robot_Actuators.md): DC 모터, 서보 모터, 스테퍼 모터, 유압/공압 시스템

## 파트 9: 로봇 프로그래밍 플랫폼 (Robotics Programming Platforms)

### 9.1. 로봇 운영체제 (Robot Operating System, ROS)
- [x] [ROS 1 vs. ROS 2: 아키텍처 차이 및 선택 가이드](./docs/09_01_01_ROS1_vs_ROS2.md)
- [x] [ROS 2 핵심 개념: 노드(Nodes), 토픽(Topics), 서비스(Services), 액션(Actions), 파라미터(Parameters)](./docs/09_01_02_ROS2_Core_Concepts.md)
- [x] [개발 환경 및 도구: Colcon 빌드 시스템, Rviz2 시각화 도구, ros2 bag 데이터 로깅](./docs/09_01_03_Development_Environment.md)
- [x] [TF (Transform Library): 로봇의 동적인 좌표 변환 관리](./docs/09_01_04_TF.md)
- [x] [URDF (Unified Robot Description Format): 로봇 모델 기술](./docs/09_01_05_URDF.md)

### 9.2. 로봇 시뮬레이터 (Robot Simulators)
- [x] [Gazebo: 물리 엔진 기반 다개체 시뮬레이션, 월드/모델 생성 및 ROS 연동](./docs/09_02_01_Gazebo.md)
- [x] [NVIDIA Isaac Sim: 포토리얼리스틱 렌더링, 물리 기반 시뮬레이션, 딥러닝용 합성 데이터 생성](./docs/09_02_02_NVIDIA_Isaac_Sim.md)
- [x] [기타 시뮬레이터: CoppeliaSim (V-REP), PyBullet, MuJoCo](./docs/09_02_03_Other_Simulators.md)

### 9.3. 임베디드 시스템 (Embedded Systems for Robotics)
- [x] [마이크로컨트롤러 (MCU): Arduino, STM32 등 펌웨어 개발](./docs/09_03_01_MCU.md)
- [x] [싱글 보드 컴퓨터 (SBC): Raspberry Pi, Jetson Nano - 리눅스 기반 로봇 제어](./docs/09_03_02_SBC.md)
- [x] [실시간 운영체제 (RTOS): FreeRTOS, Zephyr - 실시간성 보장](./docs/09_03_03_RTOS.md)

## 파트 10: AI 기반 로봇 응용 (AI-Powered Robotic Applications)

### 10.1. 자율 이동 로봇 (Autonomous Mobile Robots, AMR) & 자율주행
- [x] **[인식 (Perception)](./docs/10_01_01_Perception.md):**
  - [x] 카메라, LiDAR 데이터 처리 및 융합 (파트 4.3 CV, 파트 8.5 센서 연계)
  - [x] 센서 퓨전 심화: 칼만 필터(Kalman Filter), 확장 칼만 필터(EKF)
- [x] **[위치 추정 및 지도 작성 (Localization & Mapping)](./docs/10_01_02_Localization_and_Mapping.md):**
  - [x] SLAM (Simultaneous Localization and Mapping) 알고리즘: EKF SLAM, Particle Filter SLAM, GraphSLAM
  - [x] 주요 SLAM 기술: Visual SLAM (ORB-SLAM), LiDAR SLAM (GMapping, Cartographer)
- [x] **[경로 및 모션 계획 (Path & Motion Planning)](./docs/10_01_03_Path_and_Motion_Planning.md):**
  - [x] 전역 경로 계획 (Global Path Planning): Dijkstra, A*
  - [x] 지역 경로 계획 (Local Path Planning): Dynamic Window Approach (DWA), Timed Elastic Band (TEB)
  - [x] 샘플링 기반 계획 (Sampling-based Planning): RRT, RRT*
- [x] **[자율주행 심화](./docs/10_01_04_Advanced_Autonomous_Driving.md):**
  - [x] 행동 계획 (Behavioral Planning): 유한 상태 머신(FSM)
  - [x] 움직임 예측 (Motion Prediction)

### 10.2. 로봇 매니퓰레이션 (Robot Manipulation)
- [x] **[파지 (Grasping)](./docs/10_02_01_Grasping.md):**
  - [x] 2D/3D 객체 인식 및 자세 추정 (파트 4.3 CV 연계)
  - [x] 안정적인 파지 계획 (Stable Grasp Planning)
- [x] **[작업 계획 (Task Planning)](./docs/10_02_02_Task_Planning.md):**
  - [x] Pick-and-Place 작업 순서 계획
  - [x] 조립(Assembly)을 위한 모션 계획
- [x] **[AI 기반 제어](./docs/10_02_03_AI_based_Control.md):**
  - [x] 모방 학습 (Imitation Learning): Behavior Cloning, DAgger
  - [x] 강화학습 기반 매니퓰레이션 (파트 5 RL 연계): 희소 보상 문제, Sim-to-Real 전달

### 10.3. 항공 로보틱스 (Aerial Robotics / Drones)
- [x] [드론 동역학 및 제어 모델](./docs/10_03_01_Drone_Dynamics_and_Control.md)
- [x] [항법 (Navigation) 및 궤적 추종 (Trajectory Tracking)](./docs/10_03_02_Navigation_and_Trajectory_Tracking.md)
- [x] [비전 기반 자율 비행 (Visual Servoing)](./docs/10_03_03_Visual_Servoing.md) (파트 4.3 CV 연계)
- [x] [다중 드론 시스템 (Multi-drone Systems) 및 군집 비행 (Swarm Flight)](./docs/10_03_04_Multi_drone_Systems.md)

### 10.4. 휴머노이드 및 다리 로봇 (Humanoids and Legged Robots)
- [x] [보행 패턴 생성 (Gait Generation) 및 안정성](./docs/10_04_01_Gait_Generation.md)
- [x] [전신 동역학 및 제어 (Whole-Body Dynamics and Control)](./docs/10_04_02_Whole_Body_Control.md)
- [x] [균형 제어 (Balance Control): Zero Moment Point (ZMP)](./docs/10_04_03_Balance_Control_ZMP.md)
- [x] [강화학습 기반 보행 및 동작 생성 (파트 5 RL 연계)](./docs/10_04_04_RL_based_Motion.md)
