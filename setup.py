from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name=' Recurrent_model_for_graph_network',
    version='0.1.0',
    description='Applies RNN to graph data through Graph Network framework',
    long_description_content_type="text/markdown",
    long_description=README,
    license='GPLv2',
    packages=find_packages(),
    author='Lorenzo Jacopo Pileggi',
    author_email='l.pileggi1@studenti.unipi.it',
    keywords=['RNN', 'network analysis', 'graph network'],
    url='https://github.com/LJPileggi/Recurrent_model_for_graph_network',
)

install_requires = [
    'tensorflow-gpu==2.8.0',
    'numpy',
    'matplotlib',
    'sonnet',
    'graph_nets',
    'setuptools'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
