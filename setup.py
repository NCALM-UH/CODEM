import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="modem",
    version="0.21",
    author="Preston Hartzell, Jesse Shanahan, Bahirah Adewunmi",
    description="A package for co-registering geospatial data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bitbucket.di2e.net/scm/crrelneggs/modem.git",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7.10",
    entry_points={
        "console_scripts": [
            "modem = modem.main:main",
        ]
    },
)
