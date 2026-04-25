import argparse
from typing import Optional


def parse_arguments(available_tests: list[tuple[str, str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='Giulia Tests',
        description='Runs tests to make sure Giulia is running correctly.'
    )
    parser.add_argument(
        '-m',
        '--monitor',
        required=False,
        default=0.1,
        dest='MONITOR',
        help='System resources monitoring interval in seconds. Default: %(default)s.',
        type=float
    )
    parser.add_argument(
        '--generate',
        required=False,
        default=False,
        dest='GENERATE',
        help='Whether this run should generate precomputed test files instead of comparing them with the stored ones. Default: %(default)s.',
        type=bool
    )
    parser.add_argument(
        '--html',
        required=False,
        default=False,
        dest='OUTPUT_HTML',
        help='Whether an HTML report of the test results should be generated. Default: %(default)s.',
        type=bool
    )
    parser.add_argument(
        '--markdown',
        required=False,
        default=True,
        dest='OUTPUT_MD',
        help='Whether a Markdown report of the test results should be generated. Default: %(default)s.',
        type=bool
    )
    parser.add_argument(
        '--download-precomputed-files',
        required=False,
        default=None,
        dest='DOWNLOAD_PRECOMPUTED_FILES',
        help='Whether to download the expected precomputed files automatically or not. If not set and no results found, will ask the user for confirmation. Default: %(default)s.',
        type=Optional[bool]
    )
    parser.add_argument(
        '-t',
        '--tests',
        nargs='*',
        required=False,
        dest='TESTS',
        help='Tests to run separated by spaces. If not set, will run all tests.',
        choices=list(f'{test[0]}_{test[1]}' for test in available_tests)
    )
    return parser.parse_args()
