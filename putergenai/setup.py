from setuptools import setup, find_packages

setup(
    name='putergenai',
    version='0.1.0',
    description='A Python SDK for interacting with the Puter.js API for cloud computing and AI functionalities.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Nerve11', 
    author_email='putergenai@outlook.com',
    url='https://github.com/nerve11/putergenai',
    packages=find_packages(),
    install_requires=[
        'requests>=2.32.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    license='MIT',
    keywords='puter ai cloud chat-api \ test: filesystem txt2img img2txt txt2speech',
)
