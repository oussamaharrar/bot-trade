import os
import glob
import pandas as pd
from .evaluate import evaluate_trades

RESULTS_DIR = "results"
EVAL_FILE = os.path.join(RESULTS_DIR, "evaluation.csv")

# حذف التقييم السابق إن وجد
if os.path.exists(EVAL_FILE):
    print(f"🗑️ حذف الملف السابق: {EVAL_FILE}")
    os.remove(EVAL_FILE)

# جمع كل ملفات الصفقات
trade_files = glob.glob(os.path.join(RESULTS_DIR, "*_trades.csv"))

if not trade_files:
    print("❌ لا توجد ملفات صفقات للتقييم.")
    exit()

for path in trade_files:
    try:
        df = pd.read_csv(path)
        model_name = os.path.basename(path).replace("_trades.csv", "")
        # استخراج الفريم من اسم الملف (اختياري)
        frame = model_name.split("_")[-1] if "_" in model_name else "N/A"
        print(f"\n🔍 تقييم النموذج: {model_name} | الفريم: {frame}")
        evaluate_trades(df, model_name=model_name, save_path=EVAL_FILE, additional_info={"frame": frame})
    except Exception as e:
        print(f"⚠️ فشل تقييم {path}: {e}")
