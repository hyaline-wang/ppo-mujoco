# EMAT-mujoco

# Build Image
```
cd docker
docker build -t 'kjaebye/ma-mujoco:1.0' . 
```

# Run Container
```
docker run -it --name ma-mujoco -v ws:/root/ws --gpus=all -v /tmp/.x11-unix:/tmp/.x11-unix -e GDK_SCALE -e GDK_DPI_SCALE -p 11022:22 9d48729d6009 /bin/bash
```

# Training

## MultiHopper
1: default
```
python main.py --env_index 1 --agent_num 1
```
2:
```
python main.py --env_index 1 --agent_num 2 --batch_size 4096 --mini_batch_size 128
```
3:
```
python main.py --env_index 1 --agent_num 3 --batch_size 4096 --mini_batch_size 128
```

# Dispaly
```
python display.py 
```