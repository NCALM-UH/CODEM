import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="codem",
    version="0.22",
    author="Ognyan Moore, Preston Hartzell, Jesse Shanahan, Bahirah Adewunmi",
    description="A package for co-registering geospatial data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NCALM-UH/CODEM",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7.10",
    entry_points={
        "console_scripts": [
            "codem = codem.main:main",
        ]
    },
)
