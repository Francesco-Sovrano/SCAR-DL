#!/bin/bash

#ROOT_PATH=/public/francesco_sovrano
#PYTHON_REV=3.7
#PYTHON_VERSION=$PYTHON_REV.1
ROOT_PATH=$3
PYTHON_REV=$1
PYTHON_VERSION=$PYTHON_REV.$2
PYTHON_DIR=Python-$PYTHON_VERSION
MY_DIR="`dirname \"$0\"`"
MY_PATH="`realpath $MY_DIR`"
LOCAL_PYTHON_PATH=${ROOT_PATH}/.localpython

# Create root path and local python path
if [ ! -d $ROOT_PATH ]; then
	mkdir $ROOT_PATH
	chmod 700 $ROOT_PATH
fi
if [ ! -d $LOCAL_PYTHON_PATH ]; then
	mkdir $LOCAL_PYTHON_PATH
fi
cd $ROOT_PATH

# Install python
if [ ! -d build ]; then
	mkdir build
	cd build

	# Download python
	[ -f $PYTHON_DIR.tgz ] || wget https://www.python.org/ftp/python/$PYTHON_VERSION/$PYTHON_DIR.tgz
	tar -zxvf $PYTHON_DIR.tgz

	# Download sqlite
	[ -f sqlite-autoconf-3240000.tar.gz ] || wget https://www.sqlite.org/2018/sqlite-autoconf-3240000.tar.gz
	tar -zxvf sqlite-autoconf-3240000.tar.gz

	# Install sqlite
	cd sqlite-autoconf-3240000
	./configure --prefix=${LOCAL_PYTHON_PATH}
	make
	make install
	cd ..

	# Install python
	cd $PYTHON_DIR
	LD_RUN_PATH=${LOCAL_PYTHON_PATH}/lib configure
	LDFLAGS="-L ${LOCAL_PYTHON_PATH}/lib"
	CPPFLAGS="-I ${LOCAL_PYTHON_PATH}/include"
	LD_RUN_PATH=${LOCAL_PYTHON_PATH}/lib make
	./configure --prefix=${LOCAL_PYTHON_PATH}
	make
	make install
	cd ..

	LINE_TO_ADD="export PATH=${LOCAL_PYTHON_PATH}/bin:\$PATH"
	echo "${LINE_TO_ADD}" >> $HOME/.bash_profile
	source $HOME/.bash_profile
fi

# Create the virtual environment
cd $ROOT_PATH
if [ ! -d ".env" ]; then
	virtualenv -p $LOCAL_PYTHON_PATH/bin/python$PYTHON_REV .env
fi
