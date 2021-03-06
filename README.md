[![Build Status](https://travis-ci.org/hagne/mie-py.svg?branch=master)](https://travis-ci.org/hagne/mie-py)

# mie-py

Mie scattering calculations as described in Bohren and Huffman (1). 
Python code is based on a fortran to python translation conduced by Herbert Kaiser. Last time I checked [the link to 
the original code](http://scatterlib.googlecode.com/files/bhmie_herbert_kaiser_july2012.py) was broken.

*(1) Bohren, C.F., Huffman, D.R., 1983. Absorption and scattering of light by small particles. Wiley, 
New York. doi:10.1002/9783527618156*
## Requirements

- numpy
- pandas

### Optioinal

- matplotlib -> adds plotting capabilities
- scipy -> needed for testing


## Installation
Execute the following line in the terminal from within the folder containing the program.

``
python setup.py install
``

## Examples

- [example gist](https://gist.github.com/6762781b4f744baeeefeea4773bcb874)