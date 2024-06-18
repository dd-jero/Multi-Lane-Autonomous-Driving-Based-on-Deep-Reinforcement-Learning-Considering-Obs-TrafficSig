## 장애물과 신호등을 고려한 디지털 가상환경 모델 구현 및 심층강화학습기반 다차선 자율주행 연구

- ### Unity + ML_Agent toolkit
  1. 도로 주행 환경
  2. 차량 에이전트 구현
  3. 가상 도로 구현 (Unity Asset 활용)
  4. 차량 장애물 및 장애물 오브젝트 배치
     
- ### Python API
  1. 심층 강화학습 모델 (customed DQN)
  2. Pytorch + CuPy/NumPy + CUDA (Anaconda 사용)
  3. Customed DQN(Deep Q-Network) = Vanilla DQN + Dueling Architecture + PER(Prioritized Experience Replay) + NoisyNet
  4. Epsilon Greedy decaying, Soft target Update
     
  *추가로 고려해본 기법: Multi-step Learning, Distributional Q-learning

### 결과분석: Proposed System vs Vanilla DQN vs NoisyNet 
 - 학습의 수렴성 및 안정성 증가
 - 학습 효율성 개선
 - 차선 유지, 장애물 회피, 신호 준수 자율 주행 성공
