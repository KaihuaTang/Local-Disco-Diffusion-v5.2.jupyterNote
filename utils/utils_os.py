import subprocess, os, sys
import shutil

def gitclone(url):
    res = subprocess.run(['git', 'clone', url], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def pipi(modulestr):
    res = subprocess.run(['pip', 'install', modulestr], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def pipie(modulestr):
    res = subprocess.run(['git', 'install', '-e', modulestr], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def wget(url, outputdir):
    res = subprocess.run(['wget', url, '-P', f'{outputdir}'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def createPath(filepath):
    os.makedirs(filepath, exist_ok=True)
    

def preprocess(model_path, project_dir):
    if not os.path.exists("CLIP"):
        gitclone("https://github.com/openai/CLIP")
    sys.path.append(f'{project_dir}/CLIP')
    
    if not os.path.exists("guided-diffusion"):
        gitclone("https://github.com/crowsonkb/guided-diffusion")
    sys.path.append(f'{project_dir}/guided-diffusion')
    
    if not os.path.exists("ResizeRight"):
        gitclone("https://github.com/assafshocher/ResizeRight.git")
    sys.path.append(f'{project_dir}/ResizeRight')
    
    if not os.path.exists('pytorch3d-lite'):
        gitclone("https://github.com/MSFTserver/pytorch3d-lite.git")
    sys.path.append(f'{project_dir}/pytorch3d-lite')
    
    if not os.path.exists("disco-diffusion"):
        gitclone("https://github.com/alembics/disco-diffusion.git")
    if os.path.exists('disco_xform_utils.py') is not True:
        shutil.move('disco-diffusion/disco_xform_utils.py', 'disco_xform_utils.py')
    sys.path.append(project_dir)