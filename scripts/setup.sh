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


if [ -d $HOME/.pyenv ]; then
    rm $HOME/.pyenv -rf
fi

git clone https://github.com/pyenv/pyenv $HOME/.pyenv

idem_patch_bashprofile() {
    # idempotently modifies the user's bashprofile with the passed string.
    PROFILE=$HOME/.bash_profile
    if ! grep -Fxq "$1" $PROFILE; then
        echo "$1" >> $PROFILE
    fi
}

if [ ! -f $HOME/.bash_profile ]; then
    touch $HOME/.bash_profile
fi

idem_patch_bashprofile 'export PATH=$PATH:$HOME/.local/bin:$HOME/bin'
idem_patch_bashprofile 'export PATH=$HOME/.pyenv/shims:$PATH'
idem_patch_bashprofile 'export PATH=$HOME/.pyenv/bin:$PATH'
idem_patch_bashprofile 'eval "$(pyenv init -)"'
source ~/.bash_profile

pyenv install 3.10.6
pyenv global 3.10.6
pip install pipx
pipx ensurepath
