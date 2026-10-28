# Week 8 ESM-2 GPU server setup

This guide prepares `vcc-server-kse` for the Week 8 ESM-2 practical. It uses a separate `uv` environment and keeps datasets and outputs on the server.

The task is multi-label localization prediction. SwissProt partitions 0–3 are used for training, partition 4 for validation, and `hpa_testset.csv` is used as an external test set.

## 1. Confirm SSH access

The local `~/.ssh/config` should contain:

```sshconfig
Host vcc-server-kse
    HostName 185.9.41.6
    User olehmezh
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

From a local terminal:

```bash
ssh vcc-server-kse
```

Keep the private key on the local machine; never copy it to the server.

## 2. Check the server and GPU

```bash
hostname
whoami
nvidia-smi
```

`nvidia-smi` should show a GPU, driver and available memory. If it is unavailable, do not start ESM training yet.

## 3. Install `uv` and create an environment

On the server:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
mkdir -p ~/kse-week8/data ~/kse-week8/outputs
cd ~/kse-week8
uv venv --python 3.11
```

Install the CUDA-enabled PyTorch build and required packages:

```bash
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch>=2.11,<2.12'

uv pip install --python .venv/bin/python \
  'transformers>=4.50' 'accelerate>=1.0' \
  'numpy>=2.0' 'pandas>=2.2' 'scikit-learn>=1.5'
```

If the driver shown by `nvidia-smi` does not support CUDA 12.8, use the corresponding PyTorch CUDA index instead. Do not mix CPU and CUDA PyTorch wheels.

Smoke-test the environment:

```bash
.venv/bin/python -c "import torch, transformers; print(torch.__version__); print(transformers.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 4. Copy the script and datasets

Run these commands locally from the repository root:

```bash
ssh vcc-server-kse 'mkdir -p ~/kse-week8/data ~/kse-week8/outputs'
scp kse-github/practicals/week8/train_esm.py vcc-server-kse:~/kse-week8/
scp kse-github/practicals/week8/Swissprot_Train_Validation_dataset.csv vcc-server-kse:~/kse-week8/data/
scp kse-github/practicals/week8/hpa_testset.csv vcc-server-kse:~/kse-week8/data/
```

On the server:

```bash
cd ~/kse-week8
ls -lh train_esm.py data/
```

## 5. Run a one-epoch smoke test

This checks downloading, tokenization, data loading, GPU execution and output writing:

```bash
uv run --python .venv/bin/python train_esm.py \
  --data-path data/Swissprot_Train_Validation_dataset.csv \
  --hpa-path data/hpa_testset.csv \
  --mode frozen --max-length 256 --batch-size 2 --epochs 1 \
  --output-dir outputs/smoke_test
```

Check the result:

```bash
ls -lh outputs/smoke_test/
cat outputs/smoke_test/test_metrics.json
```

## 6. Start the frozen-backbone run

Use `tmux` so the job survives an SSH disconnect:

```bash
cd ~/kse-week8
tmux new -s esm_frozen
```

Inside `tmux`:

```bash
uv run --python .venv/bin/python train_esm.py \
  --data-path data/Swissprot_Train_Validation_dataset.csv \
  --hpa-path data/hpa_testset.csv \
  --mode frozen --max-length 400 --batch-size 8 --epochs 5 \
  --output-dir outputs/esm35m_frozen
```

Detach with `Ctrl+B`, then `D`. Reconnect with:

```bash
ssh vcc-server-kse
cd ~/kse-week8
tmux attach -t esm_frozen
```

Monitor GPU use from another terminal:

```bash
ssh vcc-server-kse
watch -n 2 nvidia-smi
```

## 7. Partial and full fine-tuning

Start these only after the frozen run works. The server script uses one explicit trainability parameter:

- `--num-unfrozen-layers 0`: frozen backbone and trainable classification head;
- `--num-unfrozen-layers N`: train the head and the last `N` Transformer blocks;
- `--num-unfrozen-layers -1`: full fine-tuning.

```bash
uv run --python .venv/bin/python train_esm.py \
  --data-path data/Swissprot_Train_Validation_dataset.csv \
  --hpa-path data/hpa_testset.csv \
  --num-unfrozen-layers 2 --learning-rate 2e-5 \
  --batch-size 4 --epochs 5 --output-dir outputs/esm35m_partial
```

```bash
uv run --python .venv/bin/python train_esm.py \
  --data-path data/Swissprot_Train_Validation_dataset.csv \
  --hpa-path data/hpa_testset.csv \
  --num-unfrozen-layers -1 --learning-rate 2e-5 --batch-size 2 --epochs 5 \
  --output-dir outputs/esm35m_full
```

If GPU memory is insufficient, reduce `--batch-size` or `--max-length`.

## 8. Copy results back

Run from the local machine:

```bash
scp vcc-server-kse:~/kse-week8/outputs/esm35m_frozen/history.csv .
scp vcc-server-kse:~/kse-week8/outputs/esm35m_frozen/test_metrics.json .
scp vcc-server-kse:~/kse-week8/outputs/esm35m_frozen/config.json .
```

Copy the checkpoint only when needed:

```bash
scp vcc-server-kse:~/kse-week8/outputs/esm35m_frozen/best_model.pt .
```

## Troubleshooting

- If `uv` is not found, run `source "$HOME/.local/bin/env"`.
- If `CUDA: False`, check `nvidia-smi` and verify that a CUDA PyTorch wheel is installed.
- For out-of-memory errors, try `--batch-size 2 --max-length 256`.
- If SSH disconnects, reconnect and run `tmux attach -t esm_frozen`; do not start a duplicate job.

## Security and storage

- Keep the private SSH key only on the local machine.
- Do not expose Jupyter or SSH ports beyond the intended access rules.
- Keep checkpoints and outputs under `~/kse-week8/outputs`.
- Remove old checkpoints when disk space becomes limited.
