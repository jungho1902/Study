# Semantic/Instance Segmentation 심화

**이미지 분할(Image Segmentation)**은 이미지를 픽셀(pixel) 단위로 분할하여 각 픽셀이 어떤 클래스에 속하는지를 예측하는 컴퓨터 비전의 핵심 과제입니다. 단순한 이미지 분류(Image Classification)나 객체 탐지(Object Detection)보다 더 정교한 수준의 이미지 이해를 필요로 합니다.

이미지 분할은 크게 **Semantic Segmentation**과 **Instance Segmentation**으로 나뉩니다.

## 1. Semantic Segmentation (의미론적 분할)

**Semantic Segmentation**은 이미지 내의 모든 픽셀을 미리 정의된 클래스(category)로 분류하는 작업입니다. 예를 들어, "사람", "자동차", "하늘", "도로"와 같이 각 픽셀이 어떤 의미를 갖는 클래스에 속하는지를 예측합니다.

- **핵심 특징:** 같은 클래스에 속하는 여러 객체들을 구분하지 않습니다. 예를 들어, 이미지에 사람이 3명 있다면, Semantic Segmentation은 이 3명을 모두 '사람'이라는 하나의 클래스로 취급하여 같은 색으로 칠합니다.
- **주요 아키텍처:**
  - **FCN (Fully Convolutional Network):**
    - 기존 CNN의 완전 연결 레이어(Fully-Connected Layer)를 합성곱 레이어(Convolutional Layer)로 대체하여, 입력 이미지의 공간 정보를 유지하면서 픽셀 단위의 예측을 가능하게 한 최초의 모델입니다.
    - 다운샘플링(Downsampling)으로 인해 손실된 해상도를 복원하기 위해 업샘플링(Upsampling) 기법인 **전치 합성곱(Transposed Convolution)** 또는 **역합성곱(Deconvolution)**을 사용합니다.
  - **U-Net:**
    - FCN을 기반으로 하며, 의학 이미지(Medical Image) 분할을 위해 개발되었습니다.
    - 인코더(Encoder) 부분에서 다운샘플링을 통해 특징을 추출하고, 디코더(Decoder) 부분에서 업샘플링을 통해 해상도를 복원하는 **대칭적인 U자형 구조**를 가집니다.
    - **Skip Connection:** 인코더의 특징 맵을 디코더의 동일한 해상도 레벨에 직접 연결하여, 다운샘플링 과정에서 손실된 저수준(low-level)의 정밀한 위치 정보를 보존합니다. 이로 인해 더 정확한 경계선(boundary) 예측이 가능합니다.
  - **DeepLab:**
    - **Atrous Convolution (Dilated Convolution):** 필터 내부에 간격(hole)을 두어, 파라미터 수를 늘리지 않으면서도 더 넓은 영역의 특징(receptive field)을 볼 수 있게 하는 기법입니다. 이를 통해 해상도 손실을 최소화하면서 고수준의 특징을 추출할 수 있습니다.
    - **CRF (Conditional Random Field):** 모델의 최종 출력에 CRF를 후처리로 적용하여, 픽셀 간의 관계를 고려함으로써 분할 결과를 더 부드럽고 의미론적으로 일관성 있게 만듭니다.

## 2. Instance Segmentation (객체 인스턴스 분할)

**Instance Segmentation**은 Semantic Segmentation에서 한 단계 더 나아간 과제입니다. 같은 클래스에 속하더라도, 각각의 개별 객체(instance)를 모두 구별하여 분할합니다.

- **핵심 특징:** 이미지에 사람이 3명 있다면, Instance Segmentation은 이들을 '사람1', '사람2', '사람3'과 같이 서로 다른 객체로 인식하고 각각 다른 색으로 칠합니다. 즉, **객체 탐지(Object Detection)와 Semantic Segmentation이 결합된 형태**라고 볼 수 있습니다.
- **주요 아키텍처:**
  - **Mask R-CNN:**
    - Instance Segmentation 분야에서 가장 대표적인 모델로, Faster R-CNN을 확장한 구조입니다.
    - **작동 방식 (2-Stage Detector):**
      1. **Region Proposal Network (RPN):** 먼저 객체가 있을 만한 후보 영역(Region of Interest, RoI)을 제안합니다. (객체 탐지)
      2. **RoIAlign:** 제안된 RoI에 대해 클래스 분류(Classification), 경계 상자 회귀(Bounding Box Regression)를 수행함과 동시에, **픽셀 단위의 마스크(Mask)를 예측하는 브랜치(branch)를 추가**합니다.
    - **RoIAlign:** 기존의 RoIPooling에서 발생하는 소수점 좌표의 양자화(quantization) 문제를 해결하여, 마스크 예측의 정확도를 크게 향상시켰습니다.
  - **YOLACT (You Only Look At CoefficienTs):**
    - 실시간 Instance Segmentation을 목표로 하는 **1-Stage Detector**입니다.
    - 마스크를 직접 예측하는 대신, 이미지 전체에 대한 '프로토타입 마스크(prototype masks)' 집합과 각 객체 인스턴스에 대한 '마스크 계수(mask coefficients)'를 예측합니다.
    - 이 두 결과를 선형 결합하여 최종 인스턴스 마스크를 생성함으로써 매우 빠른 속도를 달성합니다.

## 3. 응용 분야

- **자율 주행:** 도로, 차선, 보행자, 다른 차량 등을 정밀하게 인식하여 주행 경로를 계획하고 위험을 감지합니다.
- **의료 영상 분석:** CT나 MRI 이미지에서 종양, 장기 등의 영역을 정확하게 분할하여 의사의 진단을 보조합니다.
- **위성 이미지 분석:** 건물, 숲, 강 등을 자동으로 탐지하고 면적을 계산합니다.
- **증강 현실 (AR):** 특정 객체나 사람을 정확히 인식하여 그 위에 가상의 이미지를 덧씌웁니다.
