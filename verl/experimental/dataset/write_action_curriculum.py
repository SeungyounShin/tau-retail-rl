# Copyright 2025 Amazon.com Inc and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sized
from typing import List

import random

from omegaconf import DictConfig, ListConfig
from torch.utils.data import Sampler

from verl.experimental.dataset.sampler import AbstractCurriculumSampler


# WRITE action 이름 정의는 데이터 전처리 스크립트와 동일하게 유지합니다.
WRITE_ACTIONS = {
    "exchange_delivered_order_items",
    "cancel_pending_order",
    "return_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
}


class WriteActionCountCurriculumSampler(AbstractCurriculumSampler):
    """WRITE_ACTIONS 개수가 적은 예제부터 순서대로 제공하는 커리큘럼 샘플러.

    - 데이터셋 내부의 각 샘플에서 reward_model.ground_truth 를 읽어 WRITE_ACTIONS 개수를 계산합니다.
    - 같은 개수 그룹 내에서는 재현 가능한(시드 기반) 셔플을 수행하여 편향을 줄입니다.
    - 동적 커리큘럼: update(batch) 호출 횟수(스텝)에 따라 허용 최대 WRITE_ACTIONS를 점진적으로 증가시킵니다.
      * steps_per_phase: 각 단계가 지속되는 스텝 수(정수)
      * 또는 phase_steps: 누적 스텝 경계 리스트(예: [1000, 3000, 6000])
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        self.seed = int(data_config.get("seed", 1))
        self.shuffle_within_group: bool = bool(
            data_config.get("sampler", {}).get("shuffle_within_group", True)
        )
        # 최소 WRITE_ACTIONS 개수(기본 1). 0-write 샘플을 건너뛰고 싶을 때 사용
        self.min_write_actions: int = int(
            data_config.get("sampler", {}).get("min_write_actions", 1)
        )
        # 시작 WRITE_ACTIONS 임계치(기본 min_write_actions)
        self.start_write_actions: int = int(
            data_config.get("sampler", {}).get("start_write_actions", self.min_write_actions)
        )

        # 동적 커리큘럼 설정
        sampler_cfg = data_config.get("sampler", {})
        # 각 단계 당 스텝 수 (정수, 0이면 비활성화)
        self.steps_per_phase: int = int(sampler_cfg.get("steps_per_phase", 0))
        # 누적 스텝 경계 리스트 (예: [1000, 3000, 6000])
        raw_phase_steps = sampler_cfg.get("phase_steps", None)
        if isinstance(raw_phase_steps, ListConfig) or isinstance(raw_phase_steps, list):
            self.phase_steps: List[int] = [int(x) for x in list(raw_phase_steps)]
        else:
            self.phase_steps = []
        # 최대 허용 WRITE_ACTIONS (옵션)
        self.max_write_actions_cap = sampler_cfg.get("max_write_actions", None)
        self.max_write_actions_cap = int(self.max_write_actions_cap) if self.max_write_actions_cap is not None else None

        # 인덱스별 WRITE_ACTIONS 개수 계산 및 그룹화
        index_to_count: List[int] = self._compute_write_action_counts()
        count_to_indices: dict[int, List[int]] = defaultdict(list)
        for idx, cnt in enumerate(index_to_count):
            count_to_indices[cnt].append(idx)

        # 그룹 내 셔플(시드 고정)
        rng = random.Random(self.seed)
        for group in count_to_indices.values():
            if self.shuffle_within_group:
                rng.shuffle(group)

        self._count_to_indices = count_to_indices
        self._sorted_counts = sorted(count_to_indices.keys())
        self._max_observed_count = max(self._sorted_counts) if self._sorted_counts else 0

        # 현재 허용 최대 write 개수 초기화
        self._current_max_write = max(self.start_write_actions, self.min_write_actions)
        if self.max_write_actions_cap is not None:
            self._current_max_write = min(self._current_max_write, self.max_write_actions_cap)

        # 진행 상태
        self._steps_seen = 0

    def __iter__(self) -> Iterator[int]:
        # 현재 단계 기준으로 인덱스 순서를 구성 (개수 오름차순, 그룹 내 셔플 순서 유지)
        allowed_max = self._get_allowed_max_write()
        ordered_indices: List[int] = []
        for cnt in self._sorted_counts:
            if cnt < self.min_write_actions:
                continue
            if cnt > allowed_max:
                break
            ordered_indices.extend(self._count_to_indices[cnt])
        return iter(ordered_indices)

    def __len__(self) -> int:
        # 현재 단계에서 반환될 인덱스 수
        allowed_max = self._get_allowed_max_write()
        total = 0
        for cnt in self._sorted_counts:
            if cnt < self.min_write_actions:
                continue
            if cnt > allowed_max:
                break
            total += len(self._count_to_indices[cnt])
        return total

    def update(self, batch) -> None:
        # 스텝 카운트 증가 및 단계 업데이트
        self._steps_seen += 1
        new_allowed = self._compute_allowed_max_write(self._steps_seen)
        if new_allowed != self._current_max_write:
            self._current_max_write = new_allowed
            # 다음 에폭부터 반영됨 (num_workers=0이라 에폭 경계에서 sampler 재-iter)
        return

    # 내부 유틸
    def _compute_write_action_counts(self) -> List[int]:
        counts: List[int] = []
        dataframe = getattr(self.data_source, "dataframe", None)
        length = len(self.data_source)

        if dataframe is not None:
            for i in range(length):
                try:
                    row = dataframe[i]
                except Exception:
                    row = self.data_source[i]
                counts.append(self._count_write_actions_from_row(row))
        else:
            for i in range(length):
                row = self.data_source[i]
                counts.append(self._count_write_actions_from_row(row))

        return counts

    @staticmethod
    def _count_write_actions_from_row(row: dict) -> int:
        try:
            reward_model = row.get("reward_model", {}) if isinstance(row, dict) else {}
            gt_actions = reward_model.get("ground_truth", [])
            return sum(1 for action in gt_actions if action.get("name") in WRITE_ACTIONS)
        except Exception:
            return 0

    def _get_allowed_max_write(self) -> int:
        return self._current_max_write

    def _compute_allowed_max_write(self, steps_seen: int) -> int:
        # 1) phase_steps 리스트가 있다면 우선 사용 (누적 스텝 기준)
        if self.phase_steps:
            # 몇 개의 경계를 지났는지 = 현재 단계-1
            phase_index = 0
            while phase_index < len(self.phase_steps) and steps_seen >= int(self.phase_steps[phase_index]):
                phase_index += 1
            # 시작 임계치에서 phase_index만큼 증가
            allowed = self.start_write_actions + phase_index
        # 2) steps_per_phase가 설정되어 있으면 등간격 증가
        elif self.steps_per_phase and self.steps_per_phase > 0:
            inc = steps_seen // int(self.steps_per_phase)
            allowed = self.start_write_actions + inc
        else:
            # 동적 증가 비활성화
            allowed = self._current_max_write

        # 상한 적용: 관측된 최대/설정 최대
        allowed = min(allowed, self._max_observed_count)
        if self.max_write_actions_cap is not None:
            allowed = min(allowed, int(self.max_write_actions_cap))
        # 최소 하한 보장
        allowed = max(allowed, self.min_write_actions)
        return allowed 