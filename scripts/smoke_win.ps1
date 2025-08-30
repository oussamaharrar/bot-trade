$ErrorActionPreference = 'Stop'

param(
    [string]$Python = 'python'
)

Write-Host 'Installing package...'
& $Python -m pip install -e .

Write-Host 'CLI help checks...'
& $Python -m bot_trade.train_rl --help | Out-Null
& $Python -m bot_trade.tools.monitor_manager --help | Out-Null

# Generate tiny dataset for tests
$py = @"
import pandas as pd, numpy as np, os, time
os.makedirs('data/1m', exist_ok=True)
rows = 2000
base = int(time.time()*1000)
df = pd.DataFrame({
    'timestamp':[base+i*60000 for i in range(rows)],
    'open':np.random.rand(rows),
    'high':np.random.rand(rows)+1,
    'low':np.random.rand(rows),
    'close':np.random.rand(rows),
    'volume':np.random.rand(rows),
    'quote_volume':np.random.rand(rows),
    'num_trades':np.random.randint(1,100,size=rows),
    'taker_buy_base':np.random.rand(rows),
    'taker_buy_quote':np.random.rand(rows),
    'symbol':['BTCUSDT']*rows,
    'frame':['1m']*rows,
})
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df.to_csv('data/1m/BTCUSDT-1m-smoke.csv', index=False)
"@
$py | & $Python -

# Run tiny training job on CPU
$env:CUDA_VISIBLE_DEVICES = ''
$trainArgs = '--frame 1m --symbol BTCUSDT --device 0 --n-envs 1 --n-steps 500 --total-steps 1500 --artifact-every-steps 500 --no-monitor --data-root data --log-level ERROR'
Write-Host 'Running short training (will be interrupted)...'
$proc = Start-Process -FilePath $Python -ArgumentList "-m bot_trade.train_rl $trainArgs" -PassThru -NoNewWindow
Start-Sleep -Seconds 5
if (-not $proc.HasExited) {
    Stop-Process -Id $proc.Id
    Wait-Process -Id $proc.Id -ErrorAction SilentlyContinue
}

# Assert artefacts exist
if (-not (Test-Path 'agents/BTCUSDT/1m/deep_rl.zip')) { throw 'agent zip missing' }
if (-not (Test-Path 'results/BTCUSDT/1m/logs/train.log')) { throw 'train log missing' }

Write-Host 'Attempting resume...'
& $Python -m bot_trade.train_rl $trainArgs --resume | Out-Null

Write-Host 'Smoke OK'
