import os
import sys

# Configure Giulia into path
root_dir = os.path.dirname(__file__)
sys.path.insert(1, root_dir)

from giulia.inputs import InputConfig

if __name__ == '__main__':
    config_file = sys.argv[1] if len(sys.argv) > 1 else None

    if config_file is None:
        config_file = os.path.join(root_dir, 'config.yml')

    config = InputConfig.load(config_file)
    config.run()
