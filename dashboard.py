import os
import subprocess
import cmd
import shutil
import pandas as pd

MODEL_DIR = 'models'
RESULTS_DIR = 'results'
LOG_DIR = 'logs'
MEMORY_DIR = 'memory'
REPORTS_DIR = 'reports'
EVAL_LOG = 'model_evaluation_log.csv'

class Dashboard(cmd.Cmd):
    intro = 'Bot Dashboard. type help or ? to list commands.'
    prompt = '(bot) '

    def do_train(self, arg):
        subprocess.run(['python', 'autolearn.py'])

    def do_train_rl(self, arg):
        subprocess.run(['python', 'train_rl.py'])

    def do_run_rl(self, arg):
        subprocess.run(['python', 'run_rl_agent.py'])

    def do_eval(self, arg):
        subprocess.run(['python', 'evaluate_model.py'])

    def do_eval_full(self, arg):
        subprocess.run(['python', 'evaluate_model_metrics.py'])

    def do_reset(self, arg):
        for d in [MODEL_DIR, RESULTS_DIR, LOG_DIR, MEMORY_DIR]:
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        print('Workspace reset')

    def do_stats(self, arg):
        best_model = None
        if os.path.exists(EVAL_LOG):
            df = pd.read_csv(EVAL_LOG)
            if not df.empty:
                best_row = df.sort_values('f1_score', ascending=False).iloc[0]
                best_model = best_row.get('model_path')
                print(f"Best model: {best_model} (F1: {best_row['f1_score']:.4f})")
                print('F1 history:')
                print(df[['timestamp','f1_score']].tail())
        state_file = os.path.join(RESULTS_DIR, 'latest_state.json')
        if os.path.exists(state_file):
            with open(state_file) as f:
                import json
                state = json.load(f)
            print(f"Last PnL: {state.get('final_value')}")
        elif best_model is None:
            print('No stats available.')

    def do_exit(self, arg):
        'Exit the dashboard'
        return True

if __name__ == '__main__':
    Dashboard().cmdloop()
