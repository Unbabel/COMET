# -*- coding: utf-8 -*-
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="unbabel-comet",
    version="0.0.7",
    author="Ricardo Rei, Craig Stewart, Catarina Farinha, Alon Lavie",
    download_url="https://github.com/Unbabel/COMET",
    author_email="ricardo.rei@unbabel.com, craig.stewart@unbabel.com, catarina.farinha@unbabel.com, alon.lavie@unbabel.com",
    packages=find_packages(exclude=["tests"]),
    description="High-quality Machine Translation Evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
