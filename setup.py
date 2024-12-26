from setuptools import setup, find_packages
import os

print('find_packages:', find_packages())
setup(
    description='Monst3r as a package',
    name='dust3r',
    version='0.0',
    packages=find_packages(),
    install_requires=[
        'torch==2.0.1',
        'torchvision==0.15.2',
        'roma',
        'gradio',
        'matplotlib',
        'tqdm',
        'opencv-python',
        'scipy',
        'einops',
        'trimesh',
        'tensorboard',
        'pyglet<2',
        'huggingface-hub[torch]>=0.22',
        'websockets==13.1',
        "seaborn",
        "sam2 @ file://localhost/" + os.path.abspath("third_party/sam2"),
        "depth_pro @ file://localhost/" + os.path.abspath("third_party/ml-depth-pro"),
    ],
    extras_require={
        'all': [
            'croco @ git+https://github.com/Junyi42/croco_package.git'
        ],
    },
)