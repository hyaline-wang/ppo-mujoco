# PPO-mujoco

# Build Image
```
cd docker
docker build -t 'kjaebye/ppo-mujoco:1.0' . 
```

# Run Container
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