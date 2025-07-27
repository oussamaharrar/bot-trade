import schedule
import time
import subprocess
import os

# Interval can be set via env var e.g., '0 */6 * * *' (every 6 hours)
CRON_EXPRESSION = os.getenv("RETRAIN_CRON", "@daily")


def job():
    print("Running scheduled training...")
    subprocess.run(["python", "autolearn.py"]) 
    subprocess.run(["python", "evaluate_model.py"]) 


def main():
    if CRON_EXPRESSION == "@hourly":
        schedule.every().hour.do(job)
    elif CRON_EXPRESSION == "@daily":
        schedule.every().day.at("00:00").do(job)
    else:
        # Fallback to daily if cron not recognized
        schedule.every().day.at("00:00").do(job)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
