import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'aisha_brain'


def _kb_data_files():
    """Walk aisha_knowledge_db and return data_files entries for share/."""
    result = []
    kb_src = 'aisha_knowledge_db'
    if not os.path.isdir(kb_src):
        return result
    for dirpath, _dirs, filenames in os.walk(kb_src):
        if not filenames:
            continue
        # e.g. aisha_knowledge_db/abc123 -> share/aisha_brain/aisha_knowledge_db/abc123
        rel = os.path.relpath(dirpath, start='.')
        dest = os.path.join('share', package_name, rel)
        result.append((dest, [os.path.join(dirpath, f) for f in filenames]))
    return result


setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.json')),
    ] + _kb_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AI-SHA Team',
    maintainer_email='aisha@robot.local',
    description='AI-SHA Brain - ROS2 intent routing and knowledge system for a school robot',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'brain_node = aisha_brain.brain_node:main',
            'admin_node = aisha_brain.admin_node:main',
            'action_node = aisha_brain.action_node:main',
            'tts_node = aisha_brain.tts_node:main',
            'stt_node = aisha_brain.stt_node:main',
            'whatsapp_listener = aisha_brain.whatsapp_listener:main',
        ],
    },
)
