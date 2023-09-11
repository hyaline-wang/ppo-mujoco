# PPO-mujoco

## Use docker

### Build Image
```
cd docker
docker build -t 'kjaebye/ppo-mujoco:1.0' . 
```

### Run Container
```
docker run -it --name ppo-mujoco -v ws:/root/ws --gpus=all -v /tmp/.x11-unix:/tmp/.x11-unix -e GDK_SCALE -e GDK_DPI_SCALE -p 11022:22 "type your image id" /bin/bash
```

# Training
```
python lib/main.py --env_index 0
```
# Dispaly
```
python lib/display.py --run_dir lib/runs/......
```


# Use conda

```bash
conda create -n ppo python=3.8
conda activate ppo

pip install "cython<3"
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install -r docker/requirements.txt

```