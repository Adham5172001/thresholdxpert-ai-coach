"""Athlete Fatigue Prediction — Author: Adham Aboulkheir | ThresholdXpert"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TrainingSession:
    athlete_id: str; date: str; sport: str; duration_minutes: float
    avg_heart_rate: float; avg_power_watts: Optional[float]; tss: float; rpe: Optional[float] = None

@dataclass
class ReadinessScore:
    athlete_id: str; date: str; readiness: float; fatigue: float; form: float; recommendation: str

def compute_atl_ctl(tss_series, atl_tau=7, ctl_tau=42):
    atl = np.zeros(len(tss_series)); ctl = np.zeros(len(tss_series))
    atl_k = np.exp(-1/atl_tau); ctl_k = np.exp(-1/ctl_tau)
    for i, tss in enumerate(tss_series):
        if i == 0: atl[i] = tss; ctl[i] = tss
        else: atl[i] = atl[i-1]*atl_k + tss*(1-atl_k); ctl[i] = ctl[i-1]*ctl_k + tss*(1-ctl_k)
    return atl, ctl, ctl - atl

def generate_training_history(n_days=90, athlete_id="ATH-001", seed=42):
    np.random.seed(seed)
    from datetime import date, timedelta
    sessions = []
    start = date(2025, 1, 1)
    for i in range(n_days):
        if np.random.random() < 0.15: continue
        day = start + timedelta(days=i)
        sessions.append(TrainingSession(
            athlete_id=athlete_id, date=str(day),
            sport=np.random.choice(["cycling","running","swimming"]),
            duration_minutes=float(np.random.normal(90, 30)),
            avg_heart_rate=float(np.random.normal(155, 15)),
            avg_power_watts=float(np.random.normal(240, 40)),
            tss=float(np.clip(np.random.normal(80, 30), 20, 200)),
            rpe=float(np.clip(np.random.normal(6, 2), 1, 10))
        ))
    return sessions

class FatigueModel:
    def __init__(self, lookback=14):
        self.lookback = lookback
        self.model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        self.scaler = StandardScaler()
        self._fitted = False

    def _features(self, sessions):
        tss = [s.tss for s in sessions]; hr = [s.avg_heart_rate for s in sessions]
        atl, ctl, tsb = compute_atl_ctl(tss)
        return np.array([[atl[i], ctl[i], tsb[i], tss[i], hr[i], sessions[i].duration_minutes,
                           np.mean(tss[max(0,i-7):i+1]), np.mean(tss[max(0,i-14):i+1]),
                           np.std(tss[max(0,i-7):i+1]) if i>=1 else 0] for i in range(len(sessions))])

    def fit(self, sessions, readiness_labels):
        X = self._features(sessions)
        self.model.fit(self.scaler.fit_transform(X), readiness_labels)
        self._fitted = True; return self

    def predict_readiness(self, sessions):
        if not self._fitted: return []
        X = self._features(sessions)
        scores = np.clip(self.model.predict(self.scaler.transform(X)), 0, 100)
        tss = [s.tss for s in sessions]; atl, ctl, tsb = compute_atl_ctl(tss)
        results = []
        for i, (session, readiness) in enumerate(zip(sessions, scores)):
            fatigue = float(np.clip(atl[i], 0, 100)); form = float(tsb[i])
            if readiness >= 80: rec = "High readiness — excellent day for hard training or racing"
            elif readiness >= 60: rec = "Moderate readiness — suitable for moderate intensity"
            elif readiness >= 40: rec = "Low readiness — consider easy recovery session"
            else: rec = "Very low readiness — rest day strongly recommended"
            results.append(ReadinessScore(session.athlete_id, session.date, float(readiness), fatigue, form, rec))
        return results

if __name__ == "__main__":
    sessions = generate_training_history(n_days=90)
    tss = [s.tss for s in sessions]; atl, ctl, tsb = compute_atl_ctl(tss)
    readiness_labels = np.clip(50 + tsb*0.5 + np.random.normal(0, 5, len(sessions)), 0, 100)
    model = FatigueModel()
    model.fit(sessions, readiness_labels)
    predictions = model.predict_readiness(sessions[-7:])
    print("Last 7 days readiness:")
    for r in predictions:
        print(f"  {r.date}: Readiness={r.readiness:.0f}%, Fatigue={r.fatigue:.0f}%")
        print(f"    -> {r.recommendation[:60]}...")
