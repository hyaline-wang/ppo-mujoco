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
```
python train.py --cfg config/multi_walker2d_v0.yaml
```

# Dispaly
```
python display.py 
```