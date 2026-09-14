from setuptools import setup, find_packages

setup(
    name="enstrophy_stable_ns",
    version="0.1.0",
    author="Boris D. Andrews",
    author_email="boris.andrews@maths.ox.ac.uk",
    description="Firedrake code for the simulations in 'Strongly enstrophy-stable integrators for the incompressible Navier-Stokes equations'",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BorisAndrews/enstrophy_code",
    license="MIT",
    packages=find_packages(),
)