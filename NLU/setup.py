import setuptools

setuptools.setup(
    name="vblora",
    version="0.1.0",
    author="Yang Li, Shaobo Han, Shihao Ji",
    author_email="yli93@student.gsu.edu",
    description="PyTorch implementation of VB-LoRA.",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
