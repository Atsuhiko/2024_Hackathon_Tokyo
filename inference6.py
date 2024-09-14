from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
# from argparse import ArgumentParser
import platform

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.facerender.pirender_animate import AnimateFromCoeff_PIRender
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

def make_video(image_path, audio_path, 
                result_dir, size=256, preprocess='crop', 
                enhancer=None, background_enhancer=None):

    # parser = ArgumentParser()  
    # parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    # parser.add_argument("--source_image", default='./examples/source_image/full_body_1.png', help="path to source image")
    # parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    ref_eyeblink = None
    # parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    ref_pose = None
    # parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    checkpoint_dir = './checkpoints'
    # parser.add_argument("--result_dir", default='./results', help="path to output")
    # parser.add_argument("--pose_style", type=int, default=0,?  help="input pose style from [0, 46)")
    pose_style = 0
    # parser.add_argument("--batch_size", type=int, default=2,  help="the batch size of facerender")
    batch_size = 2
    # parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    # size = 256
    # parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    expression_scale = 1
    # parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    input_yaw = None
    # parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    input_pitch = None
    # parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    input_roll = None
    # parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    # enhancer = None
    # parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]")
    # background_enhancer = None
    # parser.add_argument("--cpu", dest="cpu", action="store_true") 
    # parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    face3dvis = False
    # parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion") 
    still = False
    # parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" ) 
    # preprocess = 'crop'
    # parser.add_argument("--verbose",action="store_true", help="saving the intermedia output or not" ) 
    verbose = True
    # parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" ) 
    old_version = False
    # parser.add_argument("--facerender", default='facevid2vid', choices=['pirender', 'facevid2vid'] ) 
    facerender = 'facevid2vid'
    
    # net structure and parameters
    # parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    net_recon = 'resnet50'
    # parser.add_argument('--init_path', type=str, default=None, help='Useless')
    #init_path = None # これが init_path()関数実行時のエラー　TypeError: 'NoneType' object is not callable の原因だった!!
    # parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    use_last_fc = False
    # parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    bfm_folder = './checkpoints/BFM_Fitting/'
    # parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')
    bfm_model = 'BFM_model_front.mat'

    # default renderer parameters
    # parser.add_argument('--focal', type=float, default=1015.)
    focal = 1015.
    # parser.add_argument('--center', type=float, default=112.)
    center = 112.
    # parser.add_argument('--camera_d', type=float, default=10.)
    camera_d = 10.
    # parser.add_argument('--z_near', type=float, default=5.)
    z_near = 5.
    # parser.add_argument('--z_far', type=float, default=15.)
    z_far = 15.

    # args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    #torch.backends.cudnn.enabled = False

    # pic_path = args.source_image
    pic_path = image_path
    #audio_path = args.driven_audio
    audio_path = audio_path
    save_dir = os.path.join(result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    # save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    pose_style = pose_style
    device = device
    batch_size = batch_size
    input_yaw_list = input_yaw
    input_pitch_list = input_pitch
    input_roll_list = input_roll
    ref_eyeblink = ref_eyeblink
    ref_pose = ref_pose

    # current_root_path = os.path.split(sys.argv[0])[0]
    current_root_path = './'
    # print(current_root_path)

    # TypeError: 'NoneType' object is not callable 解決
    sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), size, old_version, preprocess)

    #init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)
    
    if facerender == 'facevid2vid':
        animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
    elif facerender == 'pirender':
        animate_from_coeff = AnimateFromCoeff_PIRender(sadtalker_paths, device)
    else:
        raise(RuntimeError('Unknown model: {}'.format(facerender)))

    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, preprocess,\
                                                                             source_image_flag=True, pic_size=size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None

    #audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    # 3dface render
    if face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    
    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=expression_scale, still_mode=still, preprocess=preprocess, size=size, facemodel=facerender)
    
    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                enhancer=enhancer, background_enhancer=background_enhancer, preprocess=preprocess, img_size=size)
    
    shutil.move(result, save_dir+'.mp4')
    # print('The generated video is named:', save_dir+'.mp4')
    

    if not verbose:
        shutil.rmtree(save_dir)

    return save_dir+'.mp4'
