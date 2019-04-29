#!/bin/bash 

for i in flickr30k_images/*.jpg
do
	convert "$i" -resize 500x375! "flickr30k_resized/$i"
	echo "$i"
done



