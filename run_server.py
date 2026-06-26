"""ThresholdXpert AI Coach Demo — Author: Adham Aboulkheir | ThresholdXpert"""
import numpy as np, matplotlib, os, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from ml.fatigue_model import FatigueModel, generate_training_history, compute_atl_ctl

def main():
    print("ThresholdXpert AI Coach Demo")
    os.makedirs("outputs", exist_ok=True)
    sessions = generate_training_history(n_days=90, athlete_id="ATH-001")
    print(f"  {len(sessions)} training sessions over 90 days")
    tss = [s.tss for s in sessions]; atl, ctl, tsb = compute_atl_ctl(tss)
    readiness_labels = np.clip(50 + tsb*0.5 + np.random.normal(0, 5, len(sessions)), 0, 100)
    model = FatigueModel()
    model.fit(sessions, readiness_labels)
    predictions = model.predict_readiness(sessions)
    print("  Last 7 days:")
    for pred in predictions[-7:]:
        bar = "█"*int(pred.readiness/10) + "░"*(10-int(pred.readiness/10))
        print(f"  {pred.date}: [{bar}] {pred.readiness:.0f}%")
    days = np.arange(len(sessions))
    readiness_vals = [p.readiness for p in predictions]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#0d1117")
    for ax in axes: ax.set_facecolor("#161b22")
    axes[0].plot(days, atl, color="#ff7b72", linewidth=2, label="ATL (Fatigue)")
    axes[0].plot(days, ctl, color="#3fb950", linewidth=2, label="CTL (Fitness)")
    axes[0].fill_between(days, atl, ctl, where=ctl>atl, alpha=0.15, color="#3fb950")
    axes[0].set_title("ATL / CTL Training Load", color="white"); axes[0].set_xlabel("Day", color="white"); axes[0].legend(facecolor="#161b22", labelcolor="white", fontsize=8); axes[0].tick_params(colors="white"); axes[0].grid(alpha=0.3, color="#21262d")
    axes[1].plot(days, tsb, color="#00c9b1", linewidth=2)
    axes[1].fill_between(days, tsb, 0, where=tsb>0, alpha=0.2, color="#3fb950", label="Fresh")
    axes[1].fill_between(days, tsb, 0, where=tsb<=0, alpha=0.2, color="#ff7b72", label="Fatigued")
    axes[1].axhline(y=0, color="white", linestyle="--", alpha=0.3)
    axes[1].set_title("Training Stress Balance (Form)", color="white"); axes[1].set_xlabel("Day", color="white"); axes[1].legend(facecolor="#161b22", labelcolor="white", fontsize=8); axes[1].tick_params(colors="white"); axes[1].grid(alpha=0.3, color="#21262d")
    colors = ["#3fb950" if r>=70 else "#f4a261" if r>=50 else "#ff7b72" for r in readiness_vals]
    axes[2].bar(days, readiness_vals, color=colors, alpha=0.85, width=0.8)
    axes[2].axhline(y=70, color="#3fb950", linestyle="--", linewidth=1.5, label="High readiness")
    axes[2].set_title("Daily Readiness Score", color="white"); axes[2].set_xlabel("Day", color="white"); axes[2].set_ylabel("Readiness (%)", color="white"); axes[2].legend(facecolor="#161b22", labelcolor="white", fontsize=8); axes[2].tick_params(colors="white"); axes[2].grid(axis="y", alpha=0.3, color="#21262d"); axes[2].set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig("outputs/thresholdxpert_results.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Saved: outputs/thresholdxpert_results.png")
    print("  To start API: uvicorn app.main:app --reload --port 8002")

if __name__ == "__main__":
    main()
