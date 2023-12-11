from setuptools import setup

setup(name='ctrltool',
      version='0.0.1',
      author='Tomas Nagy, Ahmad Amine',
      author_email='nagytom@tier4.jp',
      url='',
      package_dir={'': 'src'},
      install_requires=['numpy>=1.24.4',
			'gpytorch>=1.11',
			'torch<=2.0.1',
			'gradient_free_optimizers>=1.3.0',
			'matplotlib>=3.5.1',
			'scipy>=1.11.1'
			]  # TODO add all of the requirements
      )
