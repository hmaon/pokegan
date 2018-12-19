#!/bin/bash


for i in unsanitized/*/*
do
	export o="${i//unsanitized/dir_per_class}"
	export o="${o//JPEG/png}"
	export o="${o//JPG/png}"
	export o="${o//jpeg/png}"
	export o="${o//jpg/png}"
	convert $i -auto-orient -alpha remove -colorspace RGB -filter Lanczos -resize 128x128 -colorspace sRGB -background white -gravity center -extent 128x128 -normalize $o
	echo $o
done
