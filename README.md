# ✔️ 연구명
### 장애물과 신호등을 고려한 디지털 가상환경 모델 구현 및 심층강화학습기반 다차선 자율주행 연구  
*Implementation of Digital Virtual Environment Model Considering Obstacles and Traffic Lights, and Research on Multi-Lane Autonomous Driving Based on Deep Reinforcement Learning*

## ✔️ 개요 
심층강화학습을 활용한 디지털 가상 도로 환경에서의 자율주행 시스템을 제안한다.  

## ✔️ 제안 시스템 

### ML-Unity 기반 디지털 시뮬레이션 환경

1. 4차선 도로 환경 구축  

| <div align="center">**투명 충돌체를 활용한 연석**</div> | <div align="center">**도로 항공뷰**</div> | 
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/8e00a6d9-8f62-4022-b99f-eb1c4bffb08c" width="450"/> | <img src="https://github.com/user-attachments/assets/f06fbfc3-5de6-4efa-ad8f-cd65bf9bf1fa" width="450"/> |  

2. 차량 장애물 및 신호등 배치 

| <div align="center">**주행 차량 장애물과 신호등**</div> | <div align="center">**정지 차량 장애물과 신호등**</div> | 
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/10a221de-dc6e-4cc9-89ff-0a943a26a1a0" width="450"/> | <img src="https://github.com/user-attachments/assets/036cff7b-d8f2-4445-9444-3112295a5707" width="450"/> |  

3. 자율주행 차량 에이전트 배치

| <div align="center">**레이 센서**</div> | <div align="center">**전방/신호 인식 카메라**</div> | 
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/d5a4292c-d91e-4809-a1f0-7217377faad0" width="450"/> | <img src="https://github.com/user-attachments/assets/5eff63da-2427-41ee-a931-d212dbeba17b" width="450"/> |  

### Custom DQN (Deep Q-Network)
- Vanilla DQN
- Dueling DQN
- Prioritized Experience Replay
- NoisyNet
- Epsilon Greedy decaying
- Soft target Update   
- 테스트해본 기법: Multi-step Learning, Distributional Q-learning 

## ✔️ 구조도
### ML-Unity 환경과 Python API 간의 데이터 통신 구조
<div align="center">
  <img src="https://github.com/user-attachments/assets/4def120a-b8e2-40c7-b327-bcdcf29611d7" alt="시스템구조1">
</div>

### 우선순위 경험 재생 기반 Deep Q-Network 학습 구조
<div align="center">
  <img src="https://github.com/user-attachments/assets/f0d2e613-373f-4090-8ad9-d365328f1c1a" alt="시스템구조2">
</div>

### Custom DQN 신경망 구조
<div align="center">
  <img src="https://github.com/user-attachments/assets/71f9c984-7854-4398-a2b9-4934a07ecef6" width= alt="시스템구조3">
</div>

## ✔️ 기술 스택 
- Language: Python, C#
- Framework: PyTorch, CuPy, NumPy, CUDA, cuDNN
- Tool: Unity, Anaconda

## ✔️ 결론 
<div align="center">
  <img src="https://github.com/user-attachments/assets/50ea6aae-d776-488b-ad64-45906a4ab8b3" width="700" alt="결과">
</div>

- Proposed System vs Vanilla DQN vs NoisyNet 
- 학습의 수렴성 및 안정성 증가 
- 학습 효율성 개선 
- 차선 유지, 장애물 회피, 신호 준수 자율 주행 성공
