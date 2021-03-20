import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="craft-text-detection",
    version="0.0.2",
    author="Clova AI Research, NAVER Corp., Ashish Jha",
    author_email="youngmin.baek@navercorp.com, arj7192@gmail.com, starcraft6723@hotmail.com",
    description="Official implementation of Character Region Awareness for Text Detection (CRAFT)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vinhtq115/CRAFT-pytorch",
    packages=['craft'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    python_requires='>=3.6',
)
