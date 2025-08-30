import os
import json
import yaml

def apply_tuning_hints_to_config(config_path="config/config.yaml"):
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    updated = False
    frames = [d for d in os.listdir("results") if d.startswith("tuning_hints_") and d.endswith(".json")]

    for file in frames:
        frame = file.replace("tuning_hints_", "").replace(".json", "")
        path = os.path.join("results", file)

        try:
            with open(path, "r", encoding="utf-8") as f:
                hints = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {file}: {e}")
            continue

        print(f"[TUNING] Applying hints for frame: {frame}")
        reward = config.setdefault("reward", {})
        risk = config.setdefault("risk", {})

        if hints.get("increase_vol_penalty"):
            reward["w_volatility"] = min(reward.get("w_volatility", 0.35) + 0.1, 1.0)
            updated = True

        if hints.get("freeze_on_drawdown"):
            risk["freeze_on_maxdd"] = True
            updated = True

        if hints.get("boost_trend_bonus"):
            reward["w_trend"] = min(reward.get("w_trend", 0.15) + 0.1, 0.6)
            updated = True

        if hints.get("penalize_mean_reversion"):
            reward["w_danger"] = min(reward.get("w_danger", 0.2) + 0.1, 1.0)
            updated = True

        if hints.get("slow_down_training"):
            config["DEFAULT_N_STEPS"] = int(config.get("DEFAULT_N_STEPS", 15000) * 0.75)
            updated = True

    if updated:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)
        print("[CONFIG ✅] Updated config.yaml based on tuning hints.")
    else:
        print("[TUNING] No updates were necessary.")

if __name__ == "__main__":
    apply_tuning_hints_to_config()
