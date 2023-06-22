from setuptools import setup, find_packages

setup(name='quadjax',
    author="Chaoyi Pan",
    author_email="chaoyip@andrew.cmu.edu",
    packages=find_packages(include="quadjax"),
    version='0.0.0',
    install_requires=[
        'gym', 
        'pandas', 
        'seaborn', 
        'matplotlib', 
        'imageio',
        'wandb', 
        'control', 
        'icecream',
        'tqdm', 
        'tyro', 
        'meshcat', 
        'sympy', 
        'gymnax',
        'jax'
        ]
)