# the main function

import gc 
import torch
import random
import numpy as np
from CLIP import clip
from datetime import datetime 
from ipywidgets import Output 
from tqdm.notebook import tqdm
from IPython import display
import matplotlib.pyplot as plt 
from PIL import Image 


from utils.utils_midas import *
from utils.utils_functions import *


normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

def do_run(args, diffusion, model,lpips_model, secondary_model, model_path, gpu_device):
    seed = args.seed
    print(range(args.start_frame, args.max_frames))

    if (args.animation_mode == "3D") and (args.midas_weight > 0.0):
        midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model(args.midas_depth_model, model_path=model_path, gpu_device=gpu_device)
    for frame_num in range(args.start_frame, args.max_frames):      
        display.clear_output(wait=True)

        # Print Frame progress if animation mode is on
        if args.animation_mode != "None":
          batchBar = tqdm(range(args.max_frames), desc ="Frames")
          batchBar.n = frame_num
          batchBar.refresh()

        
        # Inits if not video frames
        if args.animation_mode != "Video Input":
          if args.init_image in ['','none', 'None', 'NONE']:
            init_image = None
          else:
            init_image = args.init_image
          init_scale = args.init_scale
          skip_steps = args.skip_steps

        if args.animation_mode == "2D":
          if args.key_frames:
            angle = args.angle_series[frame_num]
            zoom = args.zoom_series[frame_num]
            translation_x = args.translation_x_series[frame_num]
            translation_y = args.translation_y_series[frame_num]
            print(
                f'angle: {angle}',
                f'zoom: {zoom}',
                f'translation_x: {translation_x}',
                f'translation_y: {translation_y}',
            )
          
          if frame_num > 0:
            seed += 1
            if args.resume_run and frame_num == args.start_frame:
              img_0 = cv2.imread(args.batchFolder+f"/{args.batch_name}({args.batchNum})_{args.start_frame-1:04}.png")
            else:
              img_0 = cv2.imread('prevFrame.png')
            center = (1*img_0.shape[1]//2, 1*img_0.shape[0]//2)
            trans_mat = np.float32(
                [[1, 0, translation_x],
                [0, 1, translation_y]]
            )
            rot_mat = cv2.getRotationMatrix2D( center, angle, zoom )
            trans_mat = np.vstack([trans_mat, [0,0,1]])
            rot_mat = np.vstack([rot_mat, [0,0,1]])
            transformation_matrix = np.matmul(rot_mat, trans_mat)
            img_0 = cv2.warpPerspective(
                img_0,
                transformation_matrix,
                (img_0.shape[1], img_0.shape[0]),
                borderMode=cv2.BORDER_WRAP
            )

            cv2.imwrite('prevFrameScaled.png', img_0)
            init_image = 'prevFrameScaled.png'
            init_scale = args.frames_scale
            skip_steps = args.calc_frames_skip_steps

        if args.animation_mode == "3D":
          if frame_num > 0:
            seed += 1    
            if args.resume_run and frame_num == args.start_frame:
              img_filepath = args.batchFolder+f"/{args.batch_name}({args.batchNum})_{args.start_frame-1:04}.png"
              if args.turbo_mode and frame_num > args.turbo_preroll:
                shutil.copyfile(img_filepath, 'oldFrameScaled.png')
            else:
              img_filepath = 'prevFrame.png'

            next_step_pil = do_3d_step(args, img_filepath, frame_num, midas_model, midas_transform, gpu_device)
            next_step_pil.save('prevFrameScaled.png')

            ### Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
            if args.turbo_mode:
              if frame_num == args.turbo_preroll: #start tracking oldframe
                next_step_pil.save('oldFrameScaled.png')#stash for later blending          
              elif frame_num > args.turbo_preroll:
                #set up 2 warped image sequences, old & new, to blend toward new diff image
                old_frame = do_3d_step(args, 'oldFrameScaled.png', frame_num, midas_model, midas_transform, gpu_device)
                old_frame.save('oldFrameScaled.png')
                if frame_num % int(args.turbo_steps) != 0: 
                  print('turbo skip this frame: skipping clip diffusion steps')
                  filename = f'{args.batch_name}({args.batchNum})_{frame_num:04}.png'
                  blend_factor = ((frame_num % int(args.turbo_steps))+1)/int(args.turbo_steps)
                  print('turbo skip this frame: skipping clip diffusion steps and saving blended frame')
                  newWarpedImg = cv2.imread('prevFrameScaled.png')#this is already updated..
                  oldWarpedImg = cv2.imread('oldFrameScaled.png')
                  blendedImage = cv2.addWeighted(newWarpedImg, blend_factor, oldWarpedImg,1-blend_factor, 0.0)
                  cv2.imwrite(f'{args.batchFolder}/{args.filename}',blendedImage)
                  next_step_pil.save(f'{img_filepath}') # save it also as prev_frame to feed next iteration
                  if args.vr_mode:
                    generate_eye_views(args, TRANSLATION_SCALE,args.batchFolder,filename,frame_num,midas_model, midas_transform, gpu_device)
                  continue
                else:
                  #if not a skip frame, will run diffusion and need to blend.
                  oldWarpedImg = cv2.imread('prevFrameScaled.png')
                  cv2.imwrite(f'oldFrameScaled.png',oldWarpedImg)#swap in for blending later 
                  print('clip/diff this frame - generate clip diff image')

            init_image = 'prevFrameScaled.png'
            init_scale = args.frames_scale
            skip_steps = args.calc_frames_skip_steps

        if  args.animation_mode == "Video Input":
          if not args.video_init_seed_continuity:
            seed += 1
          init_image = f'{args.videoFramesFolder}/{frame_num+1:04}.jpg'
          init_scale = args.frames_scale
          skip_steps = args.calc_frames_skip_steps

        loss_values = []
    
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    
        target_embeds, weights = [], []
        
        if args.prompts_series is not None and frame_num >= len(args.prompts_series):
          frame_prompt = args.prompts_series[-1]
        elif args.prompts_series is not None:
          frame_prompt = args.prompts_series[frame_num]
        else:
          frame_prompt = []
        
        print(args.image_prompts_series)
        if args.image_prompts_series is not None and frame_num >= len(args.image_prompts_series):
          image_prompt = args.image_prompts_series[-1]
        elif args.image_prompts_series is not None:
          image_prompt = args.image_prompts_series[frame_num]
        else:
          image_prompt = []

        print(f'Frame {frame_num} Prompt: {frame_prompt}')

        model_stats = []
        for clip_model in args.clip_models:
              cutn = 16
              model_stat = {"clip_model":None,"target_embeds":[],"make_cutouts":None,"weights":[]}
              model_stat["clip_model"] = clip_model
              
              
              for prompt in frame_prompt:
                  txt, weight = parse_prompt(prompt)
                  txt = clip_model.encode_text(clip.tokenize(prompt).to(gpu_device)).float()
                  
                  if args.fuzzy_prompt:
                      for i in range(25):
                          model_stat["target_embeds"].append((txt + torch.randn(txt.shape).device(gpu_device) * args.rand_mag).clamp(0,1))
                          model_stat["weights"].append(weight)
                  else:
                      model_stat["target_embeds"].append(txt)
                      model_stat["weights"].append(weight)
          
              if image_prompt:
                model_stat["make_cutouts"] = MakeCutouts(clip_model.visual.input_resolution, cutn, skip_augs=args.skip_augs) 
                for prompt in image_prompt:
                    path, weight = parse_prompt(prompt)
                    img = Image.open(fetch(path)).convert('RGB')
                    img = TF.resize(img, min(args.side_x, args.side_y, *img.size), T.InterpolationMode.LANCZOS)
                    batch = model_stat["make_cutouts"](TF.to_tensor(img).to(gpu_device).unsqueeze(0).mul(2).sub(1))
                    embed = clip_model.encode_image(normalize(batch)).float()
                    if args.fuzzy_prompt:
                        for i in range(25):
                            model_stat["target_embeds"].append((embed + torch.randn(embed.shape).device(gpu_device) * args.rand_mag).clamp(0,1))
                            weights.extend([weight / cutn] * cutn)
                    else:
                        model_stat["target_embeds"].append(embed)
                        model_stat["weights"].extend([weight / cutn] * cutn)
          
              model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
              model_stat["weights"] = torch.tensor(model_stat["weights"], device=gpu_device)
              if model_stat["weights"].sum().abs() < 1e-3:
                  raise RuntimeError('The weights must not sum to 0.')
              model_stat["weights"] /= model_stat["weights"].sum().abs()
              model_stats.append(model_stat)
    
        init = None
        if init_image is not None:
            init = Image.open(fetch(init_image)).convert('RGB')
            init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
            init = TF.to_tensor(init).to(gpu_device).unsqueeze(0).mul(2).sub(1)
        
        if args.perlin_init:
            if args.perlin_mode == 'color':
                init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
                init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False)
            elif args.perlin_mode == 'gray':
              init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True)
              init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
            else:
              init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
              init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
            # init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device)
            init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(gpu_device).unsqueeze(0).mul(2).sub(1)
            del init2
    
        cur_t = None
    
        def cond_fn(x, t, y=None):
            with torch.enable_grad():
                x_is_NaN = False
                x = x.detach().requires_grad_()
                n = x.shape[0]
                if args.use_secondary_model is True:
                  alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=gpu_device, dtype=torch.float32)
                  sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=gpu_device, dtype=torch.float32)
                  cosine_t = alpha_sigma_to_t(alpha, sigma)
                  out = secondary_model(x, cosine_t[None].repeat([n])).pred
                  fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                  x_in = out * fac + x * (1 - fac)
                  x_in_grad = torch.zeros_like(x_in)
                else:
                  my_t = torch.ones([n], device=gpu_device, dtype=torch.long) * cur_t
                  out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
                  fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                  x_in = out['pred_xstart'] * fac + x * (1 - fac)
                  x_in_grad = torch.zeros_like(x_in)
                for model_stat in model_stats:
                  for i in range(args.cutn_batches):
                      t_int = int(t.item())+1 #errors on last step without +1, need to find source
                      #when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                      try:
                          input_resolution=model_stat["clip_model"].visual.input_resolution
                      except:
                          input_resolution=224

                      cuts = MakeCutoutsDango(args, input_resolution,
                              Overview= args.cut_overview[1000-t_int], 
                              InnerCrop = args.cut_innercut[1000-t_int], IC_Size_Pow=args.cut_ic_pow, IC_Grey_P = args.cut_icgray_p[1000-t_int]
                              )
                      clip_in = normalize(cuts(x_in.add(1).div(2)))
                      image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                      dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                      dists = dists.view([args.cut_overview[1000-t_int]+args.cut_innercut[1000-t_int], n, -1])
                      losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                      loss_values.append(losses.sum().item()) # log loss, probably shouldn't do per cutn_batch
                      x_in_grad += torch.autograd.grad(losses.sum() * args.clip_guidance_scale, x_in)[0] / args.cutn_batches
                tv_losses = tv_loss(x_in)
                if args.use_secondary_model is True:
                  range_losses = range_loss(out)
                else:
                  range_losses = range_loss(out['pred_xstart'])
                sat_losses = torch.abs(x_in - x_in.clamp(min=-1,max=1)).mean()
                loss = tv_losses.sum() * args.tv_scale + range_losses.sum() * args.range_scale + sat_losses.sum() * args.sat_scale
                if init is not None and args.init_scale:
                    init_losses = lpips_model(x_in, init)
                    loss = loss + init_losses.sum() * args.init_scale
                x_in_grad += torch.autograd.grad(loss, x_in)[0]
                if torch.isnan(x_in_grad).any()==False:
                    grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                else:
                  # print("NaN'd")
                  x_is_NaN = True
                  grad = torch.zeros_like(x)
            if args.clamp_grad and x_is_NaN == False:
                magnitude = grad.square().mean().sqrt()
                return grad * magnitude.clamp(max=args.clamp_max) / magnitude  #min=-0.02, min=-clamp_max, 
            return grad
    
        if args.diffusion_sampling_mode == 'ddim':
            sample_fn = diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = diffusion.plms_sample_loop_progressive


        image_display = Output()
        for i in range(args.n_batches):
            if args.animation_mode == 'None':
              display.clear_output(wait=True)
              batchBar = tqdm(range(args.n_batches), desc ="Batches")
              batchBar.n = i
              batchBar.refresh()
            print('')
            display.display(image_display)
            gc.collect()
            torch.cuda.empty_cache()
            cur_t = diffusion.num_timesteps - skip_steps - 1
            total_steps = cur_t

            if args.perlin_init:
                init = regen_perlin(args, gpu_device)

            if args.diffusion_sampling_mode == 'ddim':
                samples = sample_fn(
                    model,
                    (args.batch_size, 3, args.side_y, args.side_x),
                    clip_denoised=args.clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_steps,
                    init_image=init,
                    randomize_class=args.randomize_class,
                    eta=args.eta,
                )
            else:
                samples = sample_fn(
                    model,
                    (args.batch_size, 3, args.side_y, args.side_x),
                    clip_denoised=args.clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_steps,
                    init_image=init,
                    randomize_class=args.randomize_class,
                    order=2,
                )
            
            
            # with run_display:
            # display.clear_output(wait=True)
            for j, sample in enumerate(samples):    
              cur_t -= 1
              intermediateStep = False
              if args.steps_per_checkpoint is not None:
                  if j % args.steps_per_checkpoint == 0 and j > 0:
                    intermediateStep = True
              elif j in args.intermediate_saves:
                intermediateStep = True
              with image_display:
                if j % args.display_rate == 0 or cur_t == -1 or intermediateStep == True:
                    for k, image in enumerate(sample['pred_xstart']):
                        # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                        current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
                        percent = math.ceil(j/total_steps*100)
                        if args.n_batches > 0:
                          #if intermediates are saved to the subfolder, don't append a step or percentage to the name
                          if cur_t == -1 and args.intermediates_in_subfolder is True:
                            save_num = f'{frame_num:04}' if args.animation_mode != "None" else i
                            filename = f'{args.batch_name}({args.batchNum})_{save_num}.png'
                          else:
                            #If we're working with percentages, append it
                            if args.steps_per_checkpoint is not None:
                              filename = f'{args.batch_name}({args.batchNum})_{i:04}-{percent:02}%.png'
                            # Or else, iIf we're working with specific steps, append those
                            else:
                              filename = f'{args.batch_name}({args.batchNum})_{i:04}-{j:03}.png'
                        image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                        if j % args.display_rate == 0 or cur_t == -1:
                          image.save('progress.png')
                          display.clear_output(wait=True)
                          display.display(display.Image('progress.png'))
                        if args.steps_per_checkpoint is not None:
                          if j % args.steps_per_checkpoint == 0 and j > 0:
                            if args.intermediates_in_subfolder is True:
                              image.save(f'{args.partialFolder}/{filename}')
                            else:
                              image.save(f'{args.batchFolder}/{filename}')
                        else:
                          if j in args.intermediate_saves:
                            if args.intermediates_in_subfolder is True:
                              image.save(f'{args.partialFolder}/{filename}')
                            else:
                              image.save(f'{args.batchFolder}/{filename}')
                        if cur_t == -1:
                          if frame_num == 0:
                            save_settings(args, args.batchFolder, args.batch_name, args.batchNum)
                          if args.animation_mode != "None":
                            image.save('prevFrame.png')
                          image.save(f'{args.batchFolder}/{filename}')
                          if args.animation_mode == "3D":
                            # If turbo, save a blended image
                            if args.turbo_mode and frame_num > 0:
                              # Mix new image with prevFrameScaled
                              blend_factor = (1)/int(args.turbo_steps)
                              newFrame = cv2.imread('prevFrame.png') # This is already updated..
                              prev_frame_warped = cv2.imread('prevFrameScaled.png')
                              blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1-blend_factor), 0.0)
                              cv2.imwrite(f'{args.batchFolder}/{filename}',blendedImage)
                            else:
                              image.save(f'{args.batchFolder}/{filename}')

                            if args.vr_mode:
                              generate_eye_views(args, TRANSLATION_SCALE, args.batchFolder, filename, frame_num, midas_model, midas_transform, gpu_device)

                          # if frame_num != args.max_frames-1:
                          #   display.clear_output()
            
            plt.plot(np.array(loss_values), 'r')