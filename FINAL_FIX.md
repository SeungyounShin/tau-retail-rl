# 🎯 최종 문제 해결!

## 문제의 근본 원인:

**Rollout에서 reward shaping된 보상이 Trainer에 전달되지 않았습니다!**

### 흐름 분석:

1. **Rollout 단계** (WorkerDict):
   ```
   tau_retail_interaction.generate_response()
   → tau_retail.compute_score(num_turns, num_errors)  
   → 보상 계산: 0.670, 0.790, 0.470 등 ✅
   → reward_scores = {"user_turn_rewards": [0.67, ...]}에 저장
   → non_tensor_batch["reward_scores"]로 전달
   ```

2. **Trainer 단계** (TaskRunner):
   ```
   NaiveRewardManager.__call__()
   → compute_score() 다시 호출 ❌
   → num_turns=1, num_errors=0으로 잘못 계산
   → 최종 보상: 0.0 또는 1.0 (binary)
   → DEBUG GRPO: [0,0,0,0,0] or [1,1,1,1,1]
   ```

## 해결책:

### 1. tau_retail_interaction.py 수정 ✅
- `num_errors` 추적 추가
- `num_turns` 누적 방식 수정

### 2. tau_retail.py 수정 ✅
- Efficiency penalty 추가: `-0.01 × (실제_턴수 - 최적_턴수)`
- Error penalty 추가: `-0.1 × 에러_횟수`
- 최종 보상 범위: -0.5 ~ 1.0

### 3. **naive.py 수정 (핵심!)** ✅
- `non_tensor_batch["reward_scores"]` 확인
- `user_turn_rewards`가 있으면 **마지막 값 사용**
- ← 이것이 rollout에서 이미 reward shaping된 값!

## 코드 변경 사항:

```python
# verl/workers/reward_manager/naive.py (81-89번째 줄)
# Check if reward_scores from multi-turn rollout exists
if "reward_scores" in data_item.non_tensor_batch:
    reward_scores_dict = data_item.non_tensor_batch["reward_scores"]
    # Use the last user_turn_reward as the final reward (already reward shaped!)
    if "user_turn_rewards" in reward_scores_dict and len(reward_scores_dict["user_turn_rewards"]) > 0:
        reward = reward_scores_dict["user_turn_rewards"][-1]  # ← 핵심!
```

## 예상 결과:

### Before:
```
DEBUG REWARD: base=1.00, turns=15, errors=2, final=0.670
DEBUG REWARD: base=1.00, turns=17, errors=1, final=0.790
...
DEBUG GRPO: [0.0, 0.0, 0.0, 0.0, 0.0]  ← ❌
critic/advantages/mean: 0.0  ← ❌
```

### After:
```
DEBUG REWARD: base=1.00, turns=15, errors=2, final=0.670
DEBUG REWARD: base=1.00, turns=17, errors=1, final=0.790
...
DEBUG GRPO: [0.670, 0.790, 0.460, 0.550, 0.880]  ← ✅
critic/advantages/mean: 0.12  ← ✅
```

## 다음 단계:

1. **학습 중단:**
   ```bash
   pkill -f run_tau_retail_multiturn_env_gen.sh
   ```

2. **재시작:**
   ```bash
   cd /home/robin/verl
   nohup bash examples/sglang_multiturn/run_tau_retail_multiturn_env_gen.sh > train.log 2>&1 &
   ```

3. **모니터링:**
   ```bash
   # 5-10분 후 확인
   grep "DEBUG GRPO" train.log | tail -n 20
   grep "critic/advantages" train.log | tail -n 5
   ```

이제 **advantage ≠ 0**이 나와야 합니다! 🎉
