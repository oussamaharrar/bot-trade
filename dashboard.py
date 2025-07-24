import os
import subprocess
import cmd

class Dashboard(cmd.Cmd):
    intro = 'Bot Dashboard. type help or ? to list commands.'
    prompt = '(bot) '

    def do_update_model(self, arg):
        subprocess.run(['python', 'autolearn.py'])

    def do_view_logs(self, arg):
        log_file = os.path.join('logs', 'errors.log')
        if os.path.exists(log_file):
            subprocess.run(['tail', '-n', '20', log_file])
        else:
            print('No logs found.')

    def do_generate_report(self, arg):
        subprocess.run(['python', 'evaluate_model.py'])

    def do_retrain_ai(self, arg):
        subprocess.run(['python', 'autolearn.py'])

    def do_list_models(self, arg):
        for f in sorted(os.listdir('models')):
            print(f)

    def do_edit_file(self, arg):
        if not arg:
            print('Usage: edit_file <path>')
            return
        subprocess.run(['nano', arg])

    def do_patch_line(self, arg):
        parts = arg.split()
        if len(parts) < 3:
            print('Usage: patch_line <file> <line> <text>')
            return
        file, line = parts[0], int(parts[1])
        text = ' '.join(parts[2:])
        try:
            with open(file, 'r') as f:
                lines = f.read().splitlines()
            if 1 <= line <= len(lines):
                lines[line-1] = text
                with open(file, 'w') as f:
                    f.write('\n'.join(lines) + '\n')
                print(f'Patched {file}:{line}')
            else:
                print('Line number out of range')
        except Exception as e:
            print(f'Error: {e}')

    def do_move_file(self, arg):
        parts = arg.split()
        if len(parts) != 2:
            print('Usage: move_file <src> <dest_dir>')
            return
        src, dest = parts
        os.makedirs(dest, exist_ok=True)
        try:
            os.rename(src, os.path.join(dest, os.path.basename(src)))
            print('File moved')
        except Exception as e:
            print(f'Error: {e}')

    def do_exit(self, arg):
        'Exit the dashboard'
        return True

if __name__ == '__main__':
    Dashboard().cmdloop()
