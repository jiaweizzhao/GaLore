from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="galore-torch",
    version="1.0",
    description="GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection",
    url="https://github.com/jiaweizzhao/GaLore",
    author="Jiawei Zhao",
    author_email="jiawei@caltech.edu",
    license="Apache 2.0",
    packages=["galore_torch"],
    install_requires=required,
)