import os
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

class TrendPredictor:
    def __init__(self, csv_path="data/marketing_trends.csv", strategy_info_path="data/marketing_strategies.csv", date_col=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ Fichier introuvable : {csv_path}")
        
        self.data = pd.read_csv(csv_path)
        
        # Charger le fichier contenant les noms de stratégie
        if os.path.exists(strategy_info_path):
            strategy_info = pd.read_csv(strategy_info_path)
            if 'strategy_id' in strategy_info.columns and 'strategy_name' in strategy_info.columns:
                self.strategy_names = strategy_info.set_index('strategy_id')['strategy_name'].to_dict()
            else:
                print("⚠️ Le fichier marketing_strategies.csv ne contient pas les colonnes 'strategy_id' et 'strategy_name'")
                self.strategy_names = {}
        else:
            print(f"⚠️ Fichier {strategy_info_path} non trouvé. Utilisation d'IDs comme noms.")
            self.strategy_names = {}

        # Détecter la colonne de date
        if date_col is None:
            date_candidates = [c for c in self.data.columns if c.lower() in ("date", "day", "timestamp")]
            if len(date_candidates) == 1:
                date_col = date_candidates[0]
                print(f"ℹ️ Colonne de date détectée automatiquement : '{date_col}'")
            else:
                raise ValueError("❌ Impossible de détecter une colonne de date.")
        
        self.data['date'] = pd.to_datetime(self.data[date_col])
        self.data.dropna(subset=['date'], inplace=True)

        self.strategy_industries = {sid: 'Non spécifié' for sid in self.data['strategy_id'].unique()}
        self.strategies = self.data['strategy_id'].unique()
        self.models = {}
        self.train_models()

    def train_models(self):
        print("🔄 Entraînement des modèles de tendance...")
        trained = skipped = 0
        for sid in self.strategies:
            df = self.data[self.data['strategy_id'] == sid].sort_values('date')
            if len(df) < 3:
                skipped += 1
                continue
            df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
            X = df[['days_since_start']]
            pack = {}
            for m in ['effectiveness', 'cost_efficiency', 'adoption_rate']:
                model = GradientBoostingRegressor().fit(X, df[m])
                pack[m] = model
            self.models[sid] = {'base_date': df['date'].min(), 'models': pack}
            trained += 1
        print(f"✅ Modèles entraînés pour {trained} stratégies")
        print(f"⚠️ Stratégies ignorées (données insuffisantes) : {skipped}")

    def predict(self, strategy_id, days_ahead=30):
        pack = self.models.get(strategy_id)
        if not pack:
            return None
        last_offset = (self.data['date'].max() - pack['base_date']).days
        target = last_offset + days_ahead
        X_new = pd.DataFrame({'days_since_start': [target]})
        return {m: mdl.predict(X_new)[0] for m, mdl in pack['models'].items()}

    def get_top_predicted_campaigns(self, days_ahead=30, top_n=5, min_efficiency=0.0):
        results = []
        for sid in self.models:
            now = self.predict(sid, 0)
            fut = self.predict(sid, days_ahead)
            if now is None or fut is None:
                continue
            growth = ((fut['adoption_rate'] - now['adoption_rate']) / now['adoption_rate'] * 100
                      if now['adoption_rate'] else np.nan)
            if np.isnan(growth):
                continue
            if fut['effectiveness'] < min_efficiency:
                continue
            results.append({
                'strategy_id': sid,
                'strategy_name': self.strategy_names.get(sid, str(sid)),
                'trending_week': int(round(days_ahead / 7)),
                'growth_rate': growth,
                'industry': self.strategy_industries.get(sid, 'Non spécifié')
            })
        results = sorted(results, key=lambda x: x['growth_rate'], reverse=True)
        return results[:top_n]


# ========== Lancement du script ==========

if __name__ == "__main__":
    predictor = TrendPredictor()

    print("\n🔍 Analyse des campagnes : tendances potentielles sur les 30 prochains jours\n")
    trending = predictor.get_top_predicted_campaigns(days_ahead=30, top_n=5, min_efficiency=0.0)

    if not trending:
        print("🔕 Aucune stratégie ne semble prometteuse actuellement selon les critères définis.")
    else:
        for i, camp in enumerate(trending, 1):
            name = camp['strategy_name']
            industry = camp['industry']
            growth = camp['growth_rate']
            note = "✅ Croissance attendue" if growth > 0 else "⚠️ Croissance faible ou stagnante"

            print(f"📊 Stratégie #{i} : \"{name}\"")
            print(f"   📆 Estimée tendance dans : {camp['trending_week']} semaine(s)")
            print(f"   📈 Taux d’évolution du taux d’adoption : {growth:.2f}% → {note}")
            print(f"   🌐 Industrie : {industry}")
            print("-" * 70)

        # Export CSV avec noms
        pd.DataFrame(trending).to_csv("predicted_trending_campaigns.csv", index=False)
        print("\n📁 Résultats exportés dans : 'predicted_trending_campaigns.csv'")
