# Advantage = 0 문제 디버깅 플랜

## 현재 상태
- ✅ do_sample=True 설정됨
- ✅ temperature=0.7, top_p=0.9 설정됨  
- ✅ GRPO 디버깅 코드 추가됨 (core_algos.py 309-315줄)
- ❌ advantage = 0 (여전히 발생)

## 다음 학습 실행 시 확인할 것:

### 1. GRPO 그룹별 보상 분포
```bash
# 학습 실행 후
grep "DEBUG GRPO" train.log | head -n 50
```

**예상되는 패턴:**

#### 패턴 A: 샘플링은 되지만 같은 결과 (Multi-turn 수렴)
```
[DEBUG GRPO] Group uuid-1: n=5, rewards=[1.0, 1.0, 1.0, 1.0, 1.0]
[DEBUG GRPO] Group uuid-2: n=5, rewards=[0.0, 0.0, 0.0, 0.0, 0.0]
```
→ 첫 turn은 다르지만 최종 상태는 수렴
→ **이것이 가장 가능성 높음!**

#### 패턴 B: 샘플링 작동 (정상)
```
[DEBUG GRPO] Group uuid-1: n=5, rewards=[1.0, 0.0, 1.0, 0.0, 1.0]
[DEBUG GRPO] Group uuid-2: n=5, rewards=[0.0, 1.0, 0.0, 1.0, 0.0]
```
→ 다양한 최종 결과
→ 이 경우 advantage ≠ 0 이어야 함

#### 패턴 C: 샘플링 안됨 (greedy)
```
[DEBUG GRPO] Group uuid-1: n=5, rewards=[1.0, 1.0, 1.0, 1.0, 1.0]
(모든 그룹이 완전히 동일한 값)
```
→ do_sample이 여전히 작동 안함

### 2. 패턴 A인 경우의 해결 방법:

Multi-turn task에서 binary reward를 사용하면 advantage가 0이 되기 쉽습니다.

**해결책 옵션:**

#### 옵션 1: Intermediate reward 사용
각 turn마다 partial reward를 주어 trajectory 차이를 반영:
```python
# tau_retail.py의 compute_score 함수 수정
# 최종 상태만 비교하는 대신, 중간 progress도 고려
```

#### 옵션 2: Reward shaping
```python
# 단순 0/1 대신:
reward = base_reward + diversity_bonus - error_penalty
```

#### 옵션 3: GRPO 대신 다른 알고리즘
```bash
# Best-of-N sampling or Pass@k 같은 방법 사용
algorithm.adv_estimator=grpo_passk
```

#### 옵션 4: 더 강한 샘플링
```bash
actor_rollout_ref.rollout.temperature=1.0  # 0.7 → 1.0
actor_rollout_ref.rollout.top_p=0.8        # 0.9 → 0.8 (더 제한적)
```

## 즉시 실행할 명령어:

```bash
# 1. 현재 설정 확인
cd /home/robin/verl
tail -n 100 examples/sglang_multiturn/run_tau_retail_multiturn_env_gen.sh

# 2. 학습 실행 (디버깅 포함)
bash examples/sglang_multiturn/run_tau_retail_multiturn_env_gen.sh 2>&1 | tee train.log

# 3. 실행 중 다른 터미널에서 모니터링
watch -n 5 "grep 'DEBUG GRPO' train.log | tail -n 20"

# 4. 학습 후 분석
grep "DEBUG GRPO" train.log | head -n 100 > grpo_rewards.log
grep "critic/advantages" train.log
```

## 핵심 질문:

**같은 그룹 내 5개 trajectory가 모두 같은 최종 보상을 받는 이유는?**

1. ❌ Greedy decoding (do_sample=False)
   → 이미 do_sample=True로 설정함

2. ⚠️ **Multi-turn interaction의 수렴 특성**
   → 첫 turn은 다르지만 최종 상태가 수렴
   → **가장 가능성 높음!**

3. ⚠️ Task의 특성
   → Binary reward + 쉬운/어려운 task
   → 모두 성공하거나 모두 실패

4. ⚠️ Interaction model의 deterministic 특성
   → User response가 예측 가능
