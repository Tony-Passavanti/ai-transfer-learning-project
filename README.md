# AI Transfer Learning Project

Minimal setup and run instructions for a hosted VM.

## 1. Create and activate a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Verify core imports

If your VM has restricted home-directory permissions, set a writable Matplotlib cache path first:

Linux/macOS:

```bash
export MPLCONFIGDIR=$(pwd)/.mplconfig
mkdir -p "$MPLCONFIGDIR"
```

Windows (PowerShell):

```powershell
$env:MPLCONFIGDIR = "$PWD\.mplconfig"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null
```

Then run:

```bash
python -c "import torch, torchvision, streamlit, sklearn, matplotlib, numpy, PIL"
```

## 4. Run project pipeline

```bash
python src/split_dataset.py
python src/train.py
python src/evaluate.py
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

For hosted VMs, open port `8501` in your VM/network firewall to access Streamlit remotely.
