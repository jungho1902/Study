# 5.3.4. 액터-크리틱 (Actor-Critic Methods)

**액터-크리틱(Actor-Critic)** 방법은 정책 기반 학습(Policy-Based)과 가치 기반 학습(Value-Based)을 결합한 방법입니다. 정책 경사 방법의 높은 분산(variance) 문제를 해결하면서도, 정책을 직접 학습하는 장점을 유지하기 위해 도입되었습니다.

- **액터 (Actor):** 정책(policy)을 직접 학습하며, 어떤 행동(action)을 할지를 결정합니다. (정책 기반의 역할)
- **크리틱 (Critic):** 액터가 선택한 행동이 얼마나 좋았는지를 평가(critique)합니다. 즉, 가치 함수를 학습하여 액터의 학습 과정을 돕고, 정책 업데이트의 변동성을 줄여줍니다. (가치 기반의 역할)

- **A2C (Advantage Actor-Critic) / A3C (Asynchronous Advantage Actor-Critic):** 액터-크리틱의 대표적인 알고리즘입니다. 크리틱이 Q함수 대신 어드밴티지 함수(Advantage Function, A(s,a) = Q(s,a) - V(s))를 사용하여 학습의 분산을 효과적으로 줄입니다. A3C는 여러 개의 에이전트를 병렬(asynchronous)로 실행하여 더 빠르고 안정적으로 학습하는 방식입니다.

- **DDPG (Deep Deterministic Policy Gradient):** DQN을 연속적인(continuous) 행동 공간에 적용하기 위해 개발된 오프-폴리시 액터-크리틱 알고리즘입니다. 이름처럼 결정론적 정책(deterministic policy)을 사용하며, DQN의 핵심 아이디어인 경험 재현(Experience Replay)과 타겟 네트워크(Target Network) 기법을 모두 사용합니다.

- **SAC (Soft Actor-Critic):** 탐험(Exploration)을 장려하기 위해 엔트로피(entropy) 개념을 도입한 오프-폴리시 액터-크리틱 알고리즘입니다. 보상(reward)뿐만 아니라 정책의 엔트로피까지 최대화하는 것을 목표로 하여, 더 안정적이고 효율적인 탐험을 수행합니다.
