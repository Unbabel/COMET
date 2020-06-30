# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name="unbabel-comet",
    version="0.0",
    author="Ricardo Rei, Craig Stewart, Catarina Farinha, Alon Lavie",
    download_url="https://github.com/Unbabel/COMET",
    author_email="ricardo.rei@unbabel.com, craig.stewart@unbabel.com, catarina.farinha@unbabel.com, alon.lavie@unbabel.com",
    packages=find_packages(exclude=["tests"]),
    description="Provides high-quality Machine Translation Evaluation",
    keywords=[
        "Deep Learning",
        "PyTorch",
        "Machine Translation",
        "AI",
        "NLP",
        "Evaluation",
    ],
    python_requires=">=3.6",
    setup_requires=[],
    install_requires=[
        line.strip() for line in open("requirements.txt", "r").readlines()
    ],
    entry_points={"console_scripts": ["comet = comet.cli:comet"]},
)
