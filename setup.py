from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="brain-tumor-detector",
    version="1.0.0",
    author="Brain Tumor Detection Team",
    author_email="team@braintumordetector.com",
    description="AI-powered brain MRI tumor detection and analysis system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/brain-tumor-detector",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.8.0",
            "torch-gpu>=1.11.0",
        ],
        "cloud": [
            "boto3>=1.20.0",
            "google-cloud-storage>=2.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "brain-tumor-detector=src.main:main",
            "preprocess-mri=src.data.preprocess:main",
            "train-model=src.training.train:main",
            "predict-tumor=src.inference.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt"],
    },
)
