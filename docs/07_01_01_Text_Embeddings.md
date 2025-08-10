# 텍스트 임베딩 (Text Embeddings)

텍스트 임베딩은 자연어 처리(NLP)에서 단어, 문장, 또는 문단과 같은 텍스트 데이터를 컴퓨터가 이해하고 처리할 수 있는 저차원의 연속적인 벡터(vector)로 변환하는 기술입니다. 텍스트를 벡터 공간에 표현함으로써 단어 간의 의미적, 문법적 관계를 포착할 수 있습니다.

## 1. 전통적인 단어 임베딩 (Traditional Word Embeddings)

전통적인 단어 임베딩은 각 단어에 대해 고유한 벡터를 할당합니다. 이 벡터는 해당 단어가 사용된 문맥 정보를 기반으로 학습됩니다.

### 1.1. Word2Vec

Word2Vec은 '단어의 의미는 주변 단어에 의해 결정된다'는 분포 가설(Distributional Hypothesis)에 기반한 예측 기반 임베딩 모델입니다. 2013년 Google의 Tomas Mikolov 연구팀에 의해 개발되었으며, 두 가지 주요 아키텍처가 있습니다.

- **CBOW (Continuous Bag-of-Words):** 주변 단어들(context words)을 가지고 중심 단어(center word)를 예측하는 모델입니다. 작은 데이터셋에서 더 좋은 성능을 보입니다.
- **Skip-gram:** 중심 단어를 가지고 주변 단어들을 예측하는 모델입니다. 일반적으로 CBOW보다 성능이 우수하며, 특히 희귀한 단어에 대해 더 잘 작동합니다.

Word2Vec을 통해 학습된 벡터들은 단어 간의 의미적 관계를 잘 포착합니다. 예를 들어, "king" - "man" + "woman" ≈ "queen"과 같은 벡터 연산이 가능해집니다.

### 1.2. GloVe (Global Vectors for Word Representation)

GloVe는 2014년 스탠포드 대학에서 개발한 임베딩 모델로, Word2Vec과 같은 예측 기반 모델의 장점과 LSA(Latent Semantic Analysis)와 같은 카운트 기반 모델의 장점을 결합했습니다.

- **핵심 아이디어:** 단어-단어 동시 등장 행렬(Word-Word Co-occurrence Matrix)의 전역(global) 통계 정보를 활용하여 단어 벡터를 학습합니다. 즉, 전체 말뭉치에서 두 단어가 함께 등장하는 빈도를 직접적으로 모델링에 사용합니다.
- **장점:** 대규모 말뭉치에서 더 빠르고 효율적으로 학습할 수 있으며, 단어 간의 의미적 유추(analogy) 작업에서 뛰어난 성능을 보입니다.

### 1.3. FastText

FastText는 2016년 Facebook AI Research에서 개발한 모델로, Word2Vec의 Skip-gram 모델을 확장한 것입니다.

- **핵심 차이점:** 단어를 통째로 벡터로 만드는 것이 아니라, 단어를 여러 개의 **n-gram (subword)**으로 분해하여 학습합니다. 예를 들어, "apple"이라는 단어는 "ap", "app", "ppl", "ple" 등과 같은 부분 문자열로 나뉩니다.
- **장점:**
    - **Out-of-Vocabulary (OOV) 문제 해결:** 훈련 데이터에 없던 단어가 등장하더라도, 해당 단어의 n-gram 정보를 통해 벡터를 추정할 수 있습니다.
    - **희귀 단어 처리:** 등장 빈도가 낮은 단어라도 n-gram을 다른 단어와 공유하므로 더 나은 벡터 표현을 학습할 수 있습니다.
    - 형태학적으로 유사한 단어(예: "nation", "national")에 대해 더 강건한 임베딩을 제공합니다.

## 2. 문맥적 임베딩 (Contextual Embeddings)

전통적인 단어 임베딩은 한 단어에 대해 항상 동일한 벡터를 할당하는 한계가 있습니다. 예를 들어, "bank"라는 단어는 '은행'과 '강둑'이라는 다른 의미로 사용될 수 있지만, Word2Vec에서는 동일한 벡터로 표현됩니다. 문맥적 임베딩은 이러한 동음이의어 문제를 해결하기 위해 문장 내에서 단어가 사용된 **문맥**을 고려하여 동적으로 단어 벡터를 생성합니다.

### 2.1. ELMo (Embeddings from Language Models)

ELMo는 2018년에 개발된 문맥적 임베딩의 시초로, **양방향 LSTM (bi-LSTM)** 기반의 언어 모델을 사용하여 단어 임베딩을 생성합니다.

- **작동 방식:** 문장 전체를 입력으로 받아, 순방향 LSTM과 역방향 LSTM을 모두 통과시킵니다. 각 단어의 최종 임베딩은 이 두 LSTM의 각 층(layer)에서 나온 은닉 상태(hidden state)들을 조합하여 생성됩니다.
- **특징:** 같은 단어라도 문맥에 따라 다른 벡터 값을 갖게 되어(deep contextualized word representation), 문장의 의미를 더 정교하게 파악할 수 있습니다.

### 2.2. BERT (Bidirectional Encoder Representations from Transformers)

BERT는 2018년 Google에서 발표한 모델로, 트랜스포머(Transformer)의 **인코더(Encoder)** 구조를 기반으로 합니다. ELMo와 같이 양방향 문맥을 모두 고려하지만, LSTM 대신 Self-Attention 메커니즘을 사용하여 문장 내 모든 단어의 관계를 동시에 학습합니다.

- **주요 학습 방식:**
    - **Masked Language Model (MLM):** 문장의 일부 단어를 무작위로 마스킹(masking)하고, 주변 단어들을 이용해 원래 단어를 예측하도록 학습합니다.
    - **Next Sentence Prediction (NSP):** 두 문장이 실제로 이어지는 문장인지 아닌지를 예측하도록 학습합니다.
- **특징:** 사전 훈련(pre-training)된 대규모 BERT 모델을 특정 NLP 과제(예: 분류, 개체명 인식)에 맞게 **미세 조정(fine-tuning)**하여 매우 높은 성능을 달성할 수 있습니다.

### 2.3. GPT (Generative Pre-trained Transformer)

GPT는 OpenAI에서 개발한 모델로, 트랜스포머의 **디코더(Decoder)** 구조를 기반으로 합니다. BERT와 달리, **단방향(왼쪽에서 오른쪽)** 문맥만을 고려하여 다음 단어를 예측하도록 학습됩니다.

- **작동 방식:** 이전 단어들만을 바탕으로 다음 단어를 예측하는 언어 모델링 작업을 통해 사전 훈련됩니다. 이러한 특성 때문에 텍스트 생성(text generation)과 같은 생성 과제에 매우 강력한 성능을 보입니다.
- **진화:** GPT-1에서 시작하여 GPT-2, GPT-3, GPT-4로 발전하면서 모델의 크기와 데이터 양을 대폭 늘려, 사람과 유사한 수준의 문장 생성 능력을 보여주고 있습니다. Few-shot 또는 Zero-shot 학습 능력이 뛰어나, 별도의 미세 조정 없이도 다양한 과제를 수행할 수 있습니다.
