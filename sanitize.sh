#!/bin/bash


for i in unsanitized/*/*.png
do
	convert $i -background white -alpha remove -colorspace RGB -filter Lanczos -resize 64x64\>^ -colorspace sRGB -gravity center -extent 64x64 "${i//unsanitized/dir_per_class}"
	echo $i
done
