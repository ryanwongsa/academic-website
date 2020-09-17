---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Cheatsheet for Setting up Deep Learning Cloud VMs"
subtitle: "General tips on the process to set up a deep learning vm"
summary: ""
authors: [admin]
tags: []
categories: []
date: 2020-07-16T07:45:08+01:00
lastmod: 2020-07-16T07:45:08+01:00
featured: false
draft: false
 
# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
# image:
#   caption: ""
#   focal_point: Smart
#   preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

# VM Setup Notes

My general notes / cheatsheet on how to set up deep learning environments on cloud infrastructure.

## AWS

Useful Resource: [train-deep-learning-models-on-gpus-using-amazon-ec2-spot-instance](https://aws.amazon.com/blogs/machine-learning/train-deep-learning-models-on-gpus-using-amazon-ec2-spot-instances/)

### Set up - AWS

1. Create a volume
2. Create an instance
    - Choose p3.2xlarge for V100 GPU.
    - NB: Request Spot Instance
    - set `Subnet` to same location as volume
    - In `Configure Security Group` choose select existing security group and add the `jupyter` one (need to remember how I set up this initially.)
3. Launch and Click the Connect button and copy command and run.

### Connect Volume to VM

Checks which drives are available and mounts device to folder.
```cmd
lsblk
sudo mkdir /dltraining
sudo mount /dev/xvdf /dltraining
```

### Create Snapshots

Use interface to create snapshots to move volume between subnets incase availability is limited or cost becomes too high.

## GCP

Instructions to follow when I use it again. Currently GCP preemtible VMs availability is unreliable and V100 VMs are kept alive for a few hours at a time (1-6 hours max), then shutdown due to demand most of the time.

### Set up - GCP

```cmd
export IMAGE_FAMILY=pytorch-latest-gpu
export INSTANCE_NAME="pytorch-instance"
export ZONE="europe-west4-b"
export INSTANCE_TYPE="n1-standard-8"

gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-t4,count=1" \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=500GB \
        --metadata="install-nvidia-driver=True" \
        --preemptible
```

### Other

Clone repo:

```cmd
git clone --single-branch --branch <branchname> https://<username>:<access-code>@github.com/<username>/<repo>.git
```

Start Jupyter

```cmd
jupyter notebook --ip=0.0.0.0
```

Copy files
```cmd
scp -i <pem-key> -r ubuntu@<vm-instance>:<filedir/filename> <destdir>
```

NVIDIA Apex

```cmd
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```

Tmux session start

```cmd
tmux
source activate pytorch_p36
python -m pip install -r requirements.txt
...
```

Exporting Tokens

```
export NEPTUNE_API_TOKEN="<access_token>"
export SLACK_URL="<slack_url>"
```

Autoreload notebook

```python
%load_ext autoreload
%autoreload 2
```

Profile pythong code

```python
import cProfile
cProfile.run('ds.__getitem__(15)')
```

*WIP*...
