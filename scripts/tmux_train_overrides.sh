#!/usr/bin/env bash
set -euo pipefail

SESSION="bot-train"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CFG="${ROOT}/config/config.yaml"
ENTRY="python Train_RL.py"         # عدّل لو عندك اسم مختلف
DATE_TAG="$(date +'%Y%m%d_%H%M%S')"

# خرائط التشغيل: GPU/FRAME/SEED/N_ENVS/N_STEPS/BATCH/GAMMA/LR/STACK
FRAMES=(  "1s"  "1s"  "1m"  "1m" )
GPUS=(      0     1     2     3  )
SEEDS=(   101   102   201   202 )
N_ENVS=(    32    32    24    24 )
N_STEPS=( 4096  4096  8192  8192 )
BATCH=(   4096  4096  16384 16384 )
GAMMA=(  0.999  0.999  0.99  0.99 )
LR=(    2e-4   2e-4   3e-4   3e-4 )
STACK=(     60    30     8     8 )

# أعلام إضافية للمدرب (اختياري)
EXTRA_FLAGS="--save-best --eval-interval 1000000"

# نظافة الثريدز/الذاكرة
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# مجلدات الإخراج/الأوفررايدز
OVR_DIR="${ROOT}/overrides/${DATE_TAG}"
RUNS_ROOT="${ROOT}/runs/${DATE_TAG}"
mkdir -p "${OVR_DIR}"

# أنشئ جلسة tmux
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "✋ جلسة tmux '${SESSION}' موجودة. غيّر الاسم أو انهها: tmux kill-session -t ${SESSION}"
  exit 1
fi
tmux new-session -d -s "${SESSION}" -n "init"
tmux send-keys -t "${SESSION}:0" "echo '🚀 ${SESSION} @ ${DATE_TAG}'; bash" C-m

for i in "${!FRAMES[@]}"; do
  FRAME="${FRAMES[$i]}"
  GPU="${GPUS[$i]}"
  SEED="${SEEDS[$i]}"
  NE="${N_ENVS[$i]}"
  NS="${N_STEPS[$i]}"
  BS="${BATCH[$i]}"
  GM="${GAMMA[$i]}"
  LRATE="${LR[$i]}"
  ST="${STACK[$i]}"

  WIN="${FRAME}_gpu${GPU}_s${SEED}"
  tmux new-window -t "${SESSION}" -n "${WIN}"

  RUN_DIR="${RUNS_ROOT}/${FRAME}/gpu${GPU}_seed${SEED}"
  mkdir -p "${RUN_DIR}"

  # نبني config.session.yaml بدمج الـOverrides
  CFG_OUT="${OVR_DIR}/${FRAME}_gpu${GPU}_seed${SEED}.yaml"
  python "${ROOT}/tools/make_config_override.py" \
    --base "${BASE_CFG}" \
    --out  "${CFG_OUT}" \
    --set "defaults.frame=${FRAME}" \
    --set "project.seed=${SEED}" \
    --set "rl.policy=MlpPolicy" \
    --set "rl.gamma=${GM}" \
    --set "rl.learning_rate=${LRATE}" \
    --set "rl.n_envs=${NE}" \
    --set "rl.n_steps=${NS}" \
    --set "rl.batch_size=${BS}" \
    --set "env.frame=${FRAME}" \
    --set "paths.results_dir=${RUN_DIR}" \
    --set "paths.agents_dir=agents/${FRAME}" \
    --set "paths.reports_dir=reports/${DATE_TAG}/${FRAME}/gpu${GPU}_seed${SEED}" \
    --set "indicators.windows.ao_short_window=5" \
    --set "indicators.windows.ao_long_window=34"

  # أمر التشغيل
  CMD="CUDA_VISIBLE_DEVICES=${GPU} ${ENTRY} --config '${CFG_OUT}' ${EXTRA_FLAGS}"

  tmux send-keys -t "${SESSION}:${WIN}" "cd '${ROOT}'" C-m
  tmux send-keys -t "${SESSION}:${WIN}" "echo '[LAUNCH] ${WIN} | n_envs=${NE} | n_steps=${NS} | batch=${BS} | lr=${LRATE} | gamma=${GM} | stack=${ST}'" C-m
  tmux send-keys -t "${SESSION}:${WIN}" "${CMD} 2>&1 | tee '${RUN_DIR}/train.out'" C-m
done

tmux kill-window -t "${SESSION}:init"
echo "✅ أطلقت الجلسة: ${SESSION} | attach: tmux attach -t ${SESSION}"
echo "📂 overrides: ${OVR_DIR}"
echo "📂 runs:      ${RUNS_ROOT}"
