from PIL import Image
import depth_pro
from tqdm import tqdm
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
from transformers import pipeline
def find_images(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths
# Load model and preprocessing transform
parser = argparse.ArgumentParser('ml infer', add_help=False)
parser.add_argument('--a', default=0,
                        type=int)
parser.add_argument('--b', default=1500,
                        type=int)
parser.add_argument('--dataset_name', default=None,
                        type=str)
args = parser.parse_args()
# print(args.a)
model, transform = depth_pro.create_model_and_transforms(device='cuda')
model.eval()
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf",device='cuda')
if args.dataset_name == "bonn":
  dir = '../../data/bonn/rgbd_bonn_dataset/'
elif args.dataset_name == "davis":
  dir = '../../data/davis/DAVIS/JPEGImages/480p/'
elif args.dataset_name == "sintel":
  dir = '../../data/MPI-Sintel/MPI-Sintel-training_images/training/final/'
elif args.dataset_name == "tum":
  dir = '../../data/tum/' 

for scene in tqdm(sorted(os.listdir(dir))):
  data_dir = dir + scene
  if os.path.isdir(data_dir):
    if args.dataset_name == "bonn":
      data_dir = data_dir + '/rgb_110'
    elif args.dataset_name == "tum":
      data_dir = data_dir + '/rgb_50'
    for image_path in tqdm(sorted(os.listdir(data_dir))[int(args.a):int(args.b)]):
      #print(image_path)
      if image_path.split('.')[-1]=='jpg' or image_path.split('.')[-1]=='png': 
        # depthanything v2
        image = Image.open(os.path.join(data_dir, image_path))
        depth = pipe(image)["predicted_depth"].numpy()
        #depth = prediction["depth"].cpu()  # Depth in [m].
        if args.dataset_name == "bonn":
          if not os.path.exists(data_dir.replace('rgb_110', 'rgb_110_depth_prediction_depthanything')):
            os.makedirs(data_dir.replace('rgb_110', 'rgb_110_depth_prediction_depthanything'))
          if not os.path.exists(data_dir.replace('rgb_110', 'rgb_110_depth_prediction_depthpro')):
            os.makedirs(data_dir.replace('rgb_110', 'rgb_110_depth_prediction_depthpro'))
          path_depthanything = os.path.join(data_dir, image_path).replace('rgb_110', 'rgb_110_depth_prediction_depthanything').replace('.jpg', '.npz').replace('.png', '.npz')
          path_depthpro = os.path.join(data_dir, image_path).replace('rgb_110', 'rgb_110_depth_prediction_depthpro').replace('.jpg', '.npz').replace('.png', '.npz')
        elif args.dataset_name == "tum":
          if not os.path.exists(data_dir.replace('rgb_50', 'rgb_50_depth_prediction_depthanything')):
            os.makedirs(data_dir.replace('rgb_50', 'rgb_50_depth_prediction_depthanything'))
          if not os.path.exists(data_dir.replace('rgb_50', 'rgb_50_depth_prediction_depthpro')):
            os.makedirs(data_dir.replace('rgb_50', 'rgb_50_depth_prediction_depthpro'))
          path_depthanything = os.path.join(data_dir, image_path).replace('rgb_50', 'rgb_50_depth_prediction_depthanything').replace('.jpg', '.npz').replace('.png', '.npz')
          path_depthpro = os.path.join(data_dir, image_path).replace('rgb_50', 'rgb_50_depth_prediction_depthpro').replace('.jpg', '.npz').replace('.png', '.npz')
        elif args.dataset_name == "sintel":
          if not os.path.exists(data_dir.replace('final', 'depth_prediction_depthanything')):
            os.makedirs(data_dir.replace('final', 'depth_prediction_depthanything'))
          if not os.path.exists(data_dir.replace('final', 'depth_prediction_depthpro')):
            os.makedirs(data_dir.replace('final', 'depth_prediction_depthpro'))
          path_depthanything = os.path.join(data_dir, image_path).replace('final', 'depth_prediction_depthanything').replace('.jpg', '.npz').replace('.png', '.npz')
          path_depthpro = os.path.join(data_dir, image_path).replace('final', 'depth_prediction_depthpro').replace('.jpg', '.npz').replace('.png', '.npz')
        elif args.dataset_name == "davis":
          if not os.path.exists(data_dir.replace('JPEGImages', 'depth_prediction_depthanything')):
              os.makedirs(data_dir.replace('JPEGImages', 'depth_prediction_depthanything'))
          if not os.path.exists(data_dir.replace('JPEGImages', 'depth_prediction_depthpro')):
            os.makedirs(data_dir.replace('JPEGImages', 'depth_prediction_depthpro'))
          path_depthanything = os.path.join(data_dir, image_path).replace('JPEGImages', 'depth_prediction_depthanything').replace('.jpg', '.npz').replace('.png', '.npz')
          path_depthpro = os.path.join(data_dir, image_path).replace('JPEGImages', 'depth_prediction_depthpro').replace('.jpg', '.npz').replace('.png', '.npz')
      
        np.savez_compressed(path_depthanything, depth=depth)  
        # depthpro
        image, _, f_px = depth_pro.load_rgb(os.path.join(data_dir, image_path))
        image = transform(image)
        # Run inference.
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"].cpu()  # Depth in [m].
        np.savez_compressed(path_depthpro, depth=depth, focallength_px=prediction["focallength_px"].cpu())  