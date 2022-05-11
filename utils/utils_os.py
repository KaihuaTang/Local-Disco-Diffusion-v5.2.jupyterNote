import subprocess, os, sys
import pathlib, shutil

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
    
    if not os.path.exists('MiDaS'):
        gitclone("https://github.com/isl-org/MiDaS.git")
    if not os.path.exists('MiDaS/midas_utils.py'):
        shutil.move('MiDaS/utils.py', 'MiDaS/midas_utils.py')
    if not os.path.exists(f'{model_path}/dpt_large-midas-2f21e586.pt'):
        wget("https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt", model_path)
    sys.path.append(f'{project_dir}/MiDaS')
    
    if not os.path.exists("disco-diffusion"):
        gitclone("https://github.com/alembics/disco-diffusion.git")
    if os.path.exists('disco_xform_utils.py') is not True:
        shutil.move('disco-diffusion/disco_xform_utils.py', 'disco_xform_utils.py')
    sys.path.append(project_dir)
    
    
    if os.path.exists("AdaBins") is not True:
          gitclone("https://github.com/shariqfarooq123/AdaBins.git")
    if not os.path.exists(f'{project_dir}/pretrained/AdaBins_nyu.pt'):
        createPath(f'{project_dir}/pretrained')
        wget("https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt", f'{project_dir}/pretrained')
    sys.path.append(f'{project_dir}/AdaBins')