# 심층강화학습기반 장애물과 신호등을 고려한 다차션 자율주행 연구
## 2023.12 ~ 2024.01 
## 한국정보통신학회 논문지 (JKICS: The Journal of Korean Institute of Information and Communication Sciences) 2024년 2월 예정 
- 도로 주행 환경
  1. Unity + ML_Agent toolkit
  2. 차량 에이전트 구현
  3. 가상 도로 구현 (Unity Asset 활용)
  4. 차량 장애물 및 장애물 오브젝트 배치
     
- 자율주행 모델
  1. Python + Pytorch + CuPy/NumPy + CUDA (Anaconda 사용)
  2. 심층강화학습: Customed DQN(Deep Q-Network) = Vanilla DQN + Dueling Architecture + PER(Prioritized Experience Replay) + NoisyNet
  *추가로 고려해본 기법: Multi-step Learning

# 논문 결과: Proposed System vs Vanilla DQN vs NoisyNet 
=> 학습의 수렴성 및 안정성 증가, 학습 효율성(속도) 개선, 제안하는 시스템의 목표 주행 달성(차선 주행, 장애물 회피, 신호 준수)
