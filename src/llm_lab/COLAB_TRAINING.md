# Training on Google Colab

## Prerequisites

- GitHub fine-grained token stored as a **Colab secret** named `GITHUB_TOKEN`
  (Colab sidebar → Key icon → Add secret)
- Google Drive access (cells will prompt for authorization on first run)

## Notebook cells

### Cell 1 — Install & clone

```python
import subprocess, sys
from google.colab import userdata

TOKEN = userdata.get('GITHUB_TOKEN')

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'mamba-ssm', 'causal-conv1d', 'pyyaml', 'tqdm'], check=False)

subprocess.run([
    'git', 'clone', '--depth=1',
    f'https://{TOKEN}@github.com/YOUR_USERNAME/aion.git',
    '/content/aion'
], check=True)
print("Done.")
```

### Cell 2 — Mount Drive + prepare data

```python
import sys, runpy
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')
Path('/content/drive/MyDrive/aion_checkpoints/mamba_medium').mkdir(parents=True, exist_ok=True)

sys.path.insert(0, '/content/aion/src')

sys.argv = ['cli', 'download', '--target', '/content/data/raw', '--preset', 'small']
runpy.run_module('llm_lab.cli', run_name='__main__', alter_sys=True)

sys.argv = ['cli', 'prepare', '--src-dir', '/content/data/raw', '--out-dir', '/content/data']
runpy.run_module('llm_lab.cli', run_name='__main__', alter_sys=True)

sys.argv = ['cli', 'tokenizer', '--corpus', '/content/data/corpus.txt', '--out', '/content/data/tokenizer.json']
runpy.run_module('llm_lab.cli', run_name='__main__', alter_sys=True)

print("Data ready.")
```

### Cell 3 — Keep-alive thread

Prevents Colab from disconnecting due to inactivity during long training runs.

```python
import time, threading

def keep_alive():
    while True:
        time.sleep(60)
        print(".", end="", flush=True)

threading.Thread(target=keep_alive, daemon=True).start()
print("Keep-alive started.")
```

### Cell 4 — Train

Checkpoints save **directly to Google Drive** so they survive session resets.

```python
import sys, runpy, yaml, shutil
from pathlib import Path

sys.path.insert(0, '/content/aion/src')

cfg = yaml.safe_load(Path('/content/aion/src/llm_lab/configs/mamba_medium.yaml').read_text())
cfg['train_path']     = '/content/data/splits/train.txt'
cfg['val_path']       = '/content/data/splits/val.txt'
cfg['tokenizer_path'] = '/content/data/tokenizer.json'
cfg['checkpoint_dir'] = '/content/drive/MyDrive/aion_checkpoints/mamba_medium'
Path('/content/run.yaml').write_text(yaml.dump(cfg))

# Save tokenizer + config to Drive for later use
shutil.copy2('/content/data/tokenizer.json', '/content/drive/MyDrive/aion_checkpoints/tokenizer.json')
shutil.copy2('/content/run.yaml', '/content/drive/MyDrive/aion_checkpoints/run.yaml')

sys.argv = ['cli', 'train', '--config', '/content/run.yaml']
runpy.run_module('llm_lab.cli', run_name='__main__', alter_sys=True)
```

## Notes

- Run cells in order: 1 → 2 → 3 → 4
- Cell 2 takes ~5 min (downloads ~80MB of SlimPajama data)
- Cell 4 takes ~2.5h on a T4 GPU for 3000 steps with `mamba_medium`
- Checkpoints are saved every 250 steps; only the last 2 are kept locally on Drive
- Use **T4 GPU** runtime (Runtime → Change runtime type → T4 GPU)
- Do NOT use TPU runtimes — they require special code

## Using the trained model

After training completes, checkpoints are at:
```
MyDrive/aion_checkpoints/mamba_medium/step_3000.pt
MyDrive/aion_checkpoints/tokenizer.json
MyDrive/aion_checkpoints/run.yaml
```

To generate text locally:
```bash
python -m llm_lab.cli generate \
  --checkpoint /path/to/step_3000.pt \
  --config /path/to/run.yaml \
  --prompt "Once upon a time"
```
