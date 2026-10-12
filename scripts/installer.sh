#!/bin/bash

# Initialize flag variables
install_reqs=false
install_dev_reqs=false
install_llm_reqs=false
current_env=false
no_shadowing_download=false
env_name=""

# Pinned to the validated reference environment ("giulia-no-tf", Python 3.11).
# Keep these pins in sync with scripts/installer_dependencies.sh and environment.yml.
# Notes:
#   - torch is NOT listed here: it is installed automatically as a dependency of
#     Sionna 2, together with its bundled CUDA runtime. Do not install it
#     separately, to avoid overwriting the build Sionna resolves.
#   - pyproj is a runtime dependency (used by giulia/tools/tools.py), so it is
#     part of REQ. requests and tomlkit are only needed by
#     scripts/download_shadowing_files.py and examples/giulia_EVpy.
REQ="sionna==2.0.1 numpy==2.4.6 scipy==1.17.1 pandas==3.0.3 scikit-learn==1.9.0 matplotlib==3.11.0 seaborn==0.13.2 shapely==2.1.2 geopandas==1.1.4 pyproj==3.7.2 trimesh[easy]==4.12.2 mitsuba==3.8.0 astropy==8.0.1 itur==0.4.0 pyarrow==25.0.0 PyYAML==6.0.3 psutil==7.2.2 nvidia-ml-py==13.610.43 tqdm==4.68.4 requests tomlkit rich"
DEV_REQUIREMENTS="pylint"

# Function to print usage
print_usage() {
    echo "Usage: $0 [--install-requirements] [--install-dev-requirements] [--install-llm-requirements] [--current-environment] [--no-shadowing-download] --env-name <name>"
    echo
    echo "Options:"
    echo "  --install-requirements     Install basic requirements"
    echo "  --install-dev-requirements Install development requirements"
    echo "  --install-llm-requirements Install LLM requirements"
    echo "  --current-environment      Use current environment"
    echo "  --no-shadowing-download    Don't download shadowing files"
    echo "  --env-name <name>          Create a new environment with the given name"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --install-requirements)
            install_reqs=true
            shift
            ;;
        --install-dev-requirements)
            install_dev_reqs=true
            shift
            ;;
        --install-llm-requirements)
            install_llm_reqs=true
            shift
            ;;
        --current-environment)
            current_env=true
            shift
            ;;
        --no-shadowing-download)
            no_shadowing_download=true
            shift
            ;;
        --env-name)
            if [[ -n $2 ]]; then
                env_name=$2
                shift 2
            else
                echo "Error: --env-name requires a value"
                print_usage
                exit 1
            fi
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            print_usage
            exit 1
            ;;
    esac
done

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPT_PATH=$(dirname "$SCRIPT")

if [[ $install_reqs == true ]]; then
    INSTALL_DEPS=true
else
    # Ask whether to install requirements
    echo "Do you want to automatically install all the required packages? (yes/no)"
    read -r INSTALL_REQ

    # validate answer
    if [ "$INSTALL_REQ" == "no" ]; then
        INSTALL_DEPS=false
    elif [ "$INSTALL_REQ" == "yes" ]; then
        INSTALL_DEPS=true
    else
        echo "Invalid choice. Exiting the script."
        exit 1
    fi
fi

if [[ $install_dev_reqs == true ]]; then
    INSTALL_DEV_DEPS=true
else
    # Ask whether to install the development requirements
    echo "Do you want to automatically install all the development dependencies? (yes/no)"
    read -r INSTALL_DEV

    # validate answer
    if [ "$INSTALL_DEV" == "no" ]; then
        INSTALL_DEV_DEPS=false
    elif [ "$INSTALL_DEV" == "yes" ]; then
        INSTALL_DEV_DEPS=true
    else
        echo "Invalid choice. Exiting the script."
        exit 1
    fi
fi


if [[ $current_env == false && $env_name == '' ]]; then
    # Ask if the user wants to create a new environment or use the current one
    echo "Do you want to create a new Conda environment or use the current one? (new/current)"
    read -r ENV_CHOICE
elif [[ $env_name != '' ]]; then
    ENV_CHOICE="new"
else
    ENV_CHOICE="current"
fi

if [ "$ENV_CHOICE" == "new" ]; then
    if [[ $env_name == '' ]]; then
        # Ask the user for the environment name
        echo "Enter the name of the Conda environment:"
        read -r ENV_NAME
    else
        echo "Creating new Conda environment: $env_name"
        ENV_NAME=$env_name
    fi

    # Create the environment with Python 3.11
    conda create -n "$ENV_NAME" python=3.11 #-c conda-forge -y

    # Activate the environment using the bash-compatible command
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME" 
    echo "New environment '$ENV_NAME' created."
elif [ "$ENV_CHOICE" == "current" ]; then
    # Activate the bash-compatible command
    source "$(conda info --base)/etc/profile.d/conda.sh"
    echo "Using the current Conda environment."
    echo "Environment updated."
else
    echo "Invalid choice. Exiting the script."
    exit 1
fi

# Install the required packages (torch and its CUDA runtime are pulled in
# automatically by Sionna 2; do not install torch separately)
if [ $INSTALL_DEPS ]; then
    echo "Installing requirements..."
    pip install $REQ
else
    echo "Won't install the required packages."
fi
if [ $INSTALL_DEV_DEPS ]; then
    echo "Installing development requirements..."
    pip install $DEV_REQUIREMENTS
else
    echo "Won't install the development packages."
fi


# Download shadowing files
if [[ $no_shadowing_download == false ]]; then
    python "$SCRIPT_PATH/download_shadowing_files.py"
else
    echo "Won't download shadowing files as per user-preference."
fi


# Eliminate caches
pip cache purge
conda clean --all -y

# # Run the Python test scripts with specific names
# tests_passed=true
# declare -a test_scripts=("test_GPU_SimpleTrainNN_Torch.py" "Comparison_GPUandCPU_BigModel_Torch.py")
# for test_script in "${test_scripts[@]}"
# do
#     python "$SCRIPT_PATH/$test_script"
#     if [ $? -ne 1 ]; then
#         tests_passed=false
#         echo "Test failed: $test_script did not pass."
#         break
#     fi
# done

# # Check if all tests passed
# echo " "
# if [ "$tests_passed" == true ]; then
#     echo "All test passed!"
#     echo "TERMINATED SUCCESSFULLY!"
# else
#     echo "Test failed: please check the environment setup."
# fi
# fi
