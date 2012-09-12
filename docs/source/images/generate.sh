#!/bin/sh

FILES=model01

for F in $FILES
do
	pdflatex $F
	pdfcrop $F.pdf $F.pdf
	convert -density 70 $F.pdf $F.png
done
