from setuptools import setup

setup(
    name='diff2x',
    version='0.1.0',    
    description='Fast and memory-efficient upscaling without artifacts',
    url='https://github.com/peterwilli/Diff2X',
    author="Peter Willemsen",
    author_email="peter@codebuffet.co",
    license='Apache License',
    packages=['diff2x'],
    install_requires=[
        'torch>=1.6',
        'torchvision',
        'numpy',
        'pandas',
        'tqdm',
        'lmdb',
        'pillow',
        'tensorboardx',
        'wandb',
        'fasteners',
        'throttle'
    ],
    classifiers=[
        'Development Status :: 2 - Testing',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache License',  
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
)