#!/usr/bin/env bash
FONT_DL_DIR="/tmp"
FONT_OUT_DIR="fonts/cjk"
NOTO_CJK="https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJK.ttc.zip"
ATOM_CJK="https://github.com/jmlntw/atom-fonts-cjk/archive/master.zip"

# 1. Download all fonts.
if [ ! -f $FONT_DL_DIR/a.zip ]
then
    wget -O $FONT_DL_DIR/a.zip -c -- $NOTO_CJK
fi
if [ ! -f $FONT_DL_DIR/b.zip ]
then
    wget -O $FONT_DL_DIR/b.zip -c -- $ATOM_CJK
fi

# 2. Extract only the fonts into current directory without preserving
# subdirectory structure and without overwriting files.
unzip -j -n $FONT_DL_DIR/a.zip "*.[o|t]tf" -d $FONT_OUT_DIR
unzip -j -n $FONT_DL_DIR/b.zip "*.[o|t]tf" -d $FONT_OUT_DIR
for font in 6 7 10 24 25 27 35 74 75
do
  wget -P $FONT_OUT_DIR http://www.clearchinese.com/images/fonts/HDZB_$font.TTF
done
