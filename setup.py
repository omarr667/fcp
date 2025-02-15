from setuptools import setup, find_packages

setup(
    name='fcp',
    version='0.0.6',
    description="Un paquete de utilidades para el curso de diplamado de finanzas con Python de la Facultad de Ciencias (UNAM)",  
    author="Facultad de ciencias",  
    author_email="omarr667@gmail.com", 
    url="https://github.com/omarr667/fcp",
    packages=find_packages(include=["fcp", "fcp.*"]),
    include_package_data=True,  
    package_data={
        "fcp": [
            "fcp_data/*.db",  
            "fcp_data/*.json"  
        ]
    },
    install_requires=[
        "pandas", "yfinance", "numpy", "matplotlib", "scipy"
    ],
)
