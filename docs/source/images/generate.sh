#!/bin/sh

pdflatex model01

pdfcrop model01.pdf model01.pdf

convert -density 300 model01.pdf model01.png
