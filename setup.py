from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'pilla_rl_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*'))
        ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vish',
    maintainer_email='vish@wychar.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_policy_node = pilla_rl_ros.rl_policy_node:main',
            'genesis_sim_node = pilla_rl_ros.genesis_sim_refactored_node:main',
            'teleop_node = pilla_rl_ros.teleop_node:main'
        ],
    },
)
