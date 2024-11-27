from setuptools import setup, find_packages

setup(
    name='fcp',
    version='0.0.1',
    description='Paquete de funciones para el curso de Finanzas Cuantitativas con Python',
    packages=find_packages(include=["fcp", "fcp.*"]),
    include_package_data=True,  
    package_data={
        "fcp": [
            "fcp_data/*.db",  
            "fcp_data/*.json"  
        ]
    },
    install_requires=[
        "pandas", "yfinance", "numpy", "matplotlib", "scipy", "sqlite3"
    ],
)
