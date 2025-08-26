# ===== select_best_model.py =====
"""
يقرأ ملف evaluation.csv ويختار النموذج الأفضل حسب معيار محدد (مثل sharpe أو profit)
ويطبع تفاصيله ويخزن اسمه في agents/best_model.txt
"""

import pandas as pd
import os

EVAL_FILE = "results/evaluation.csv"
BEST_FILE = "agents/best_model.txt"

if not os.path.exists(EVAL_FILE):
    print("❌ ملف evaluation.csv غير موجود. قم بتشغيل التقييم أولًا.")
    exit()

# قراءة النتائج
try:
    df = pd.read_csv(EVAL_FILE)
except Exception as e:
    print(f"⚠️ فشل في قراءة الملف: {e}")
    exit()

# التحقق من وجود أعمدة مطلوبة
if "model" not in df or "sharpe" not in df:
    print("❌ الملف لا يحتوي على أعمدة كافية للتقييم.")
    exit()

# اختيار النموذج الأعلى حسب sharpe ratio
best_row = df.sort_values("sharpe", ascending=False).iloc[0]
best_model = best_row["model"]

# حفظ اسم النموذج الأفضل
os.makedirs("agents", exist_ok=True)
with open(BEST_FILE, "w") as f:
    f.write(os.path.join("agents", f"{best_model}.zip"))

print("\n🏆 أفضل نموذج حسب Sharpe Ratio:")
print(best_row.T.round(4))
print(f"\n✅ تم حفظ المسار في {BEST_FILE}")
