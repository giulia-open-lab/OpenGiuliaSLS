import os
import sys
import zipfile

import requests

url = 'https://158.42.160.122:8443/s/WRWrJfLXMZG8YzA/download'

scripts_dir = os.path.dirname(__file__)
root_dir = os.path.join(scripts_dir, '..')
shadowing_dir = os.path.join(root_dir, 'shadowing')
output_file = os.path.join(shadowing_dir, 'shadowing.zip')


def download():
    with open(output_file, "wb") as f:
        print("Downloading %s" % output_file)
        response = requests.get(url, stream=True, verify=False)
        if response.status_code != 200:
            raise ValueError(f'Status code unexpected ({response.status_code}).')
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()

    print('Download complete.')


def decompress():
    print('Decompressing...')

    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(root_dir)

    print('Decompression complete. Shadowing files ready.')


if __name__ == '__main__':
    try:
        download()
        decompress()
    except Exception as error:
        print('Could not complete download')
        print(error)
        sys.exit(1)
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)
