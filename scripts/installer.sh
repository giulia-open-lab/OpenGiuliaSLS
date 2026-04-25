#!/bin/bash

# Initialize flag variables
install_reqs=false
install_dev_reqs=false
install_llm_reqs=false
current_env=false
no_shadowing_download=false
env_name=""

DEV_REQUIREMENTS="pylint pyproj==3.7.0"

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
    conda create -n "$ENV_NAME" python=3.11 -c conda-forge -y

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

# See DEPENDENCIES.md for more information about the specifics of the following installations.
# TLDR; Torch 2.7.1, TensorFlow 2.17.0, CUDA 12.8, cuDNN 9.3

# Install TensorFlow with CUDA support
# See available versions: https://pypi.org/project/tensorflow/#history
pip install "tensorflow[and-cuda]==2.17.0" 

# Install Torch 2.7.1 with CUDA 12.8
# Note that it has to be installed before 
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 

# Install CUDA NVCC
# Use Nvcc for CUDA 12.8 as per TensorFlow requirement.
# See available versions: https://anaconda.org/nvidia/cuda-nvcc
conda install nvidia/label/cuda-12.8.1::cuda-nvcc -y 

# Install cuDNN
# Use cuDNN 9.7.1 as per TensorFlow requirement.
# See available versions: https://developer.nvidia.com/rdp/cudnn-archive
# FIXME: This downgrades sqlite 3.45.3-h5eee18b_0 --> 3.31.1-h7b6447c_0
conda install conda-forge::cudnn=9.7.1 -y 
# conda install conda-forge::sqlite=3.45.3 -y 



# Install the required packages
if [ $INSTALL_DEPS ]; then
    echo "Installing requirements..."
    pip install -r "$SCRIPT_PATH/requirements.txt" 
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
# declare -a test_scripts=("test_GPU_SimpleTrainNN_Tensorflow.py" "test_GPU_SimpleTrainNN_Torch.py" "Comparison_GPUandCPU_BigModel_Torch.py" "Comparison_GPUandCPU_BigModel_TensorFlow.py")
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
