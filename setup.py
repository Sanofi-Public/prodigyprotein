from setuptools import setup, find_packages

setup(
    name="prodigyprotein",
    author="Dr. Matthew Massett",
    author_email="matthew.massett@sanofi.com",
    maintainer="Dr. Matthew Massett",
    version="1.0.0",
    keywords=["Protein Engineering", "Directed Evolution", "Prodigy"],
    license_files=["LICENSE.md"],
    description="Protein Language based mutations",
    install_requires=[
        "tensorflow >= 2.13.0rc0, <2.16;platform_system=='Darwin'",
        "tensorflow >= 2.13, <2.16;platform_system=='Linux'",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.9",
    packages=find_packages(),
)
