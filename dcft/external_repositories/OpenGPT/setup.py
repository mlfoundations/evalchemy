import setuptools
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

with open("./README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="opengpt",
    version="0.0.5",
    author="w-is-h",
    author_email="w.kraljevic@gmail.com",
    description="OpenGPT a framework for producing grounded domain specific LLMs, and NHS-LLM a conversational model for healthcare made using OpenGPT.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cogstack/opengpt",
    packages=['opengpt'],
    install_requires=[
        'datasets>=2,<3',
        'transformers>=4.2,<5',
        'tiktoken>=0.3.2',
        'pandas',
        'openai',
        'numpy',
        'tqdm',
        'python-box',
        'jsonpickle',
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
