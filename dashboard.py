import os
import subprocess
import cmd
import shutil
import json
import pandas as pd

class Dashboard(cmd.Cmd):
    intro = 'Bot Dashboard. type help or ? to list commands.'
    prompt = '(bot) '

    # --- Core workflow commands ---
    def do_train(self, arg):
        """Train the ML model."""
        subprocess.run(['python', 'autolearn.py'])

    def do_train_rl(self, arg):
        """Train the reinforcement learning agent."""
        subprocess.run(['python', 'train_rl.py'])

    def do_run_rl(self, arg):
        """Run the reinforcement learning agent."""
        subprocess.run(['python', 'run_rl_agent.py'])

    def do_eval(self, arg):
        """Run standard evaluation."""
        subprocess.run(['python', 'evaluate_model.py'])

    def do_eval_full(self, arg):
        """Run extended evaluation metrics."""
        subprocess.run(['python', 'evaluate_model_metrics.py'])

    # --- Maintenance commands ---
    def do_reset(self, arg):
        """Wipe models/, results/, logs/ and memory/ directories."""
        dirs = ['models', 'results', 'logs', 'memory']
        for d in dirs:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        print('Workspace reset complete.')

    def do_stats(self, arg):
        """Show best model, last PnL and F1 score history."""
        best_model = 'N/A'
        f1_history = []
        log_path = 'model_evaluation_log.csv'
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            if not df.empty:
                best_row = df.sort_values('f1_score', ascending=False).iloc[0]
                best_model = best_row['model_path']
                f1_history = df['f1_score'].tolist()

        last_pnl = 'N/A'
        state_path = 'latest_state.json'
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                run_file = os.path.join('results', state.get('last_run_file', ''))
                if os.path.exists(run_file):
                    df_run = pd.read_csv(run_file)
                    if 'total_value' in df_run.columns:
                        last_pnl = df_run['total_value'].iloc[-1] - df_run['total_value'].iloc[0]
                elif 'final_value' in state:
                    last_pnl = state['final_value']
            except Exception:
                pass

        print(f"Best model: {best_model}")
        print(f"Last PnL: {last_pnl}")
        if f1_history:
            print('F1 history:', ', '.join(f"{f:.4f}" for f in f1_history))
        else:
            print('F1 history: N/A')

    def do_exit(self, arg):
        'Exit the dashboard'
        return True

if __name__ == '__main__':
    Dashboard().cmdloop()
