# sb3/callbacks.py
import os
import numpy as np
from typing import Any, Dict, List
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback,
    StopTrainingOnNoModelImprovement
)
from configs.hvac_config import TRACK_BAND

class InfoLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # 로그할 보상 컴포넌트 목록
        self.reward_keys_to_log = [
            "R_prog", "R_level", "R_fair", "R_track", "R_hum", "R_co2",
            "R_act_d", "R_cool_align", "R_energy", "R_safety", "R_dir"
        ]

    def _on_step(self) -> bool:
        infos: List[Dict[str, Any]] = self.locals.get("infos", [])
        if not infos:
            return True

        # 데이터를 수집할 리스트 초기화
        dtn_list, dtn_abs_list, frac_in_band_list = [], [], []
        comfort_list, power_list = [], []
        # 보상 컴포넌트용 딕셔너리
        reward_values = {key: [] for key in self.reward_keys_to_log}

        for info in infos:
            # 환경 관련 지표
            sr = info.get("sensor_readings", {})
            rb = info.get("reward_breakdown", {})
            temps = np.asarray(sr.get("temperatures", []), dtype=float)
            T_eff = np.asarray(rb.get("T_eff", []), dtype=float)
            if temps.size and T_eff.size:
                dtn = (temps - T_eff) / max(float(TRACK_BAND), 1e-6)
                dtn_list.append(float(np.mean(dtn)))
                dtn_abs_list.append(float(np.mean(np.abs(dtn))))
                frac_in_band = float(np.mean(np.abs(dtn) <= 1.0))
                frac_in_band_list.append(frac_in_band)

            # 쾌적도 및 전력
            comfort = info.get("comfort_data", {}).get("average_comfort")
            if comfort is not None:
                comfort_list.append(float(comfort))
            # 'step_power_consumption' -> 'step_power_W' 로 키 수정
            power = info.get("hardware_states", {}).get("step_power_W")
            if power is not None:
                power_list.append(float(power))

            # 보상 컴포넌트
            for key in self.reward_keys_to_log:
                if key in rb:
                    reward_values[key].append(float(rb[key]))

        # 평균값 계산 및 로깅
        def _log_mean(key: str, values: list):
            if values:
                self.logger.record(key, float(np.mean(values)))

        _log_mean("env/deltaT_norm_mean", dtn_list)
        _log_mean("env/abs_deltaT_norm_mean", dtn_abs_list)
        _log_mean("env/frac_within_band", frac_in_band_list)
        _log_mean("comfort/avg", comfort_list)
        _log_mean("power/step_watt", power_list)

        for key, values in reward_values.items():
            _log_mean(f"reward/{key}", values)

        return True

def build_callbacks(eval_env, cfg):
    ckpt_dir = cfg["train"]["checkpoint_path"]
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=cfg["train"]["save_freq"],
        save_replay_buffer=True,
        save_path=ckpt_dir,
        name_prefix="sac"
    )

    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,
        verbose=1
    )

    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=ckpt_dir,
        log_path=ckpt_dir,
        eval_freq=cfg["train"]["eval_freq"],
        n_eval_episodes=cfg["train"]["n_eval_episodes"],
        deterministic=True,
        callback_after_eval=stop_cb
    )

    return CallbackList([InfoLogger(), checkpoint_cb, eval_cb])