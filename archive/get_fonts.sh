#!/usr/bin/env bash
FONT_DL_PATH="tmp/fonts.zip"

# 1. Download all Google fonts.
if [ ! -f $FONT_DL_PATH ]
then
    mkdir -p tmp
    wget -O $FONT_DL_PATH -c -- https://github.com/google/fonts/archive/master.zip
fi

# 2. Extract only TrueType fonts into current directory without preserving
# subdirectory structure and without overwriting files.
unzip -j -n $FONT_DL_PATH "*.ttf" -d fonts/
