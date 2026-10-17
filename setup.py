from glob import glob

from setuptools import setup

package_name = 'person_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.*')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/models', glob('models/*.onnx') + glob('models/*.engine')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Adam Herold',
    maintainer_email='herolada@fel.cvut.cz',
    description='YOLOX-tiny person detection node.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'person_detector = person_detection.person_detector:main',
        ],
    },
)
