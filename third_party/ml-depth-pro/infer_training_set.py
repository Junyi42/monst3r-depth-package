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
            if file.lower().endswith(('rgb.jpg', 'rgb.png')):
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

if args.dataset_name == "Tartanair":
  dir = '../../data/Tartanair_proc/'
elif args.dataset_name == "spring":
  dir = '../../data/spring_proc/train/'
elif args.dataset_name == "SceneFlow":
  dir = '../../data/SceneFlow/'
elif args.dataset_name == "Vkitti":
  dir = '../../data/vkitti_2.0.3_proc/' 
elif args.dataset_name == "PointOdyssey":
  dir = '../../data/PointOdyssey_proc/' 

image_paths = find_images(dir)
for image_path in tqdm(sorted(image_paths)[int(args.a):int(args.b)]):
  # depthanything v2
  image = Image.open(image_path)
  depth = pipe(image)["predicted_depth"].numpy()
  #depth = prediction["depth"].cpu()  # Depth in [m].
  metadata = np.load(image_path.replace('_rgb.jpg', '_metadata.npz'))
  intrinsics = np.float32(metadata['camera_intrinsics'])
  focallength_px = intrinsics[0][0]
  np.savez_compressed(image_path[:-4]+'_pred_depth_depthanything', depth=depth,focallength_px=focallength_px)
  # depthpro
  image, _, f_px = depth_pro.load_rgb(image_path)
  image = transform(image)
  # Run inference.
  prediction = model.infer(image, f_px=f_px)
  depth = prediction["depth"].cpu()  # Depth in [m].
  np.savez_compressed(image_path[:-4]+'_pred_depth_depthpro', depth=depth, focallength_px=prediction["focallength_px"].cpu())  