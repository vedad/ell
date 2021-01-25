import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="elle", # Replace with your own username
    version="0.0.1",
    author="Vedad Kunovac Hodzic",
    author_email="vxh710@bham.ac.uk",
    description="Utilities and models for the reloaded Rossiter-McLaughlin method.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vedad/elle",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
