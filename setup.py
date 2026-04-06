"""
Setup configuration for nfl-live-odds-microstructure.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nfl-live-odds-microstructure",
    version="0.1.0",
    author="Isaiah Choi",
    author_email="isaiah.j.choi.27@dartmouth.edu",
    description="In-play NFL odds movements modeled as an order-book microstructure problem.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/isaiahchoi/nfl-live-odds-microstructure",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    entry_points={
        "console_scripts": [
            "nfl-odds=src.data_ingestion.play_by_play:main",
        ],
    },
)
