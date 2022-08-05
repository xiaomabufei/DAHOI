# DAHOI
DAHOI:Dynamic Anchor for Human-Object Interaction Detection

<img src="img/overall architecture of DAHOI.png"  width="800"/>

## Requirements

- PyTorch >= 1.5.1
- torchvision >= 0.6.1
- loguru (log training process and env info)
  - tabulate (format log info)

```bash
pip install -r requirements.txt
```

- Compiling CUDA operators

```bash
cd ./models/ops
sh ./make.sh
# test
python test.py
```
