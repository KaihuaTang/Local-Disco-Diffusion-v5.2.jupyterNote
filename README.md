# Local-Disco-Diffusion-v5.2.jupyterNote
A custom Disco Diffusion v5.2 that runs on local GPUS.

## Install the Requirement
```bash
###################################
###  Step by Step Installation   ##
###################################

# 1. create and activate conda environment
conda create -n disco_diffusion pip python=3.7
source activate disco_diffusion

# 2. install jupyter notebook
conda install jupyter notebook

# 3. generate jupyter config
jupyter notebook --generate-config

# 4. open "~/.jupyter/jupyter_notebook_config.py" and add the following three lines
c.NotebookApp.ip = 'YOUR_IP'
c.NotebookApp.open_browser=True
c.NotebookApp.password_required=True

# 5. set up the password of your jupyter notebook
jupyter notebook password

# 6. install conda packages
conda install pandas regex matplotlib

# 7. install pip packages
pip install opencv-python
```