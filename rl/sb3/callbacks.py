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

    def _on_step(self) -> bool:
        infos: List[Dict[str, Any]] = self.locals.get("infos", [])
        if not infos: 
            return True
        # 벡터 환경에서 여러 env의 info가 들어옴
        dtn_list, dtn_abs_list, frac_in_band_list = [], [], []
        comfort_list, power_list = [], []
        rtrack_list, rdir_list = [], []

        for info in infos:
            sr = info.get("sensor_readings", {})
            rb = info.get("reward_breakdown", {})
            temps = np.asarray(sr.get("temperatures", []), dtype=float)
            T_eff = np.asarray(rb.get("T_eff", []), dtype=float)
            if temps.size and T_eff.size:
                dtn = (temps - T_eff) / max(float(TRACK_BAND), 1e-6)
                dtn_list.append(float(np.mean(dtn)))
                dtn_abs_list.append(float(np.mean(np.abs(dtn))))
                frac_in_band = float(np.mean(np.abs(dtn) <= 1.0))  # |ΔT_norm|<=1
                frac_in_band_list.append(frac_in_band)
            comfort = info.get("comfort_data", {}).get("average_comfort", None)
            if comfort is not None:
                comfort_list.append(float(comfort))
            power = info.get("hardware_states", {}).get("step_power_consumption", None)
            if power is not None:
                power_list.append(float(power))
            if "R_track" in rb: rtrack_list.append(float(rb["R_track"]))
            if "R_dir"   in rb: rdir_list.append(float(rb["R_dir"]))

        # 평균값만 기록
        def _m(x): 
            return float(np.mean(x)) if x else None

        kv = {
            "env/deltaT_norm_mean": _m(dtn_list),
            "env/abs_deltaT_norm_mean": _m(dtn_abs_list),
            "env/frac_within_band": _m(frac_in_band_list),
            "comfort/avg": _m(comfort_list),
            "power/step_watt": _m(power_list),
            "reward/R_track": _m(rtrack_list),
            "reward/R_dir": _m(rdir_list),
        }
        for k, v in kv.items():
            if v is not None:
                self.logger.record(k, v)
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
