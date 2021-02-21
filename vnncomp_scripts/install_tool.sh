#!/bin/bash
# example install_tool.sh script for VNNCOMP for simple_adversarial_generator (https://github.com/stanleybak/simple_adversarial_generator) 
# Stanley Bak, Feb 2021

TOOL_NAME=simple_adv_gen
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

echo "Installing $TOOL_NAME"
DIR=$(dirname $(dirname $(realpath $0)))

apt-get install -y python3 python3-pip &&
apt-get install -y psmisc && # for killall, used in prepare_instance.sh script

pip3 install -r "$DIR/requirements.txt"
