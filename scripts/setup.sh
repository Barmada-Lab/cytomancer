#!/bin/bash
#
set -e

TURBO=/nfs/turbo/umms-sbarmada
SCRATCH=/scratch/sbarmada_root/sbarmada0/$USER

if [ ! -e $HOME/turbo ]; then
    ln -s $TURBO $HOME/turbo
fi

if [ ! -e $HOME/scratch ]; then
    ln -s $SCRATCH $HOME/scratch
fi

if [ ! -e $HOME/Desktop/turbo ]; then
    ln -s $TURBO $HOME/Desktop/turbo
fi

if [ ! -e $HOME/Desktop/scratch ]; then
    ln -s $SCRATCH $HOME/Desktop/scratch
fi

if [ ! -e $HOME/Desktop/ImageJ2.desktop ]; then
    cp $TURBO/shared/Fiji.app/ImageJ2.desktop $HOME/Desktop
fi

curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12

idem_patch_bashprofile() {
    # idempotently modifies the user's bashprofile with the passed string.
    PROFILE=$HOME/.bash_profile
    if ! grep -Fxq "$1" $PROFILE; then
        echo "$1" >> $PROFILE
    fi
}

if [ ! -f $HOME/.bash_profile ]; then
touch $HOME/.bash_profile
echo "if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi" >> $HOME/.bash_profile
fi

idem_patch_bashprofile 'export CYTOMANCER_MODELS_DIR=/nfs/turbo/umms-sbarmada/shared/models'
idem_patch_bashprofile 'export CYTOMANCER_COLLECTIONS_DIR=/nfs/turbo/umms-sbarmada/shared/collections'
source ~/.bash_profile

uv tool install git+https://github.com/Barmada-Lab/cytomancer
