# Introduction
The package is designed for regression analysis based on the calculation of the multiplicative indicator used earlier for geochemical exploration. It represents a mathematical calculation of the results of chemical analysis of the samples: in the numerator – the products of positive indicators of the required deposit; in the denominator—products of elements of neutral or negative indicators of the required deposit in chemical properties to the numerator elements (Grigoryan, Solovov et al., 1983; Bensman,, 1999).
The package was used by the authors to forecast placer potential (Chefranov, Lalomov et al., 2023), where its name comes from, but can be used to solve any problem of regression analysis.

# Getting Started
1. Download and install Python 3
2. Download and install git
3. Create a directory on your workstation into which to clone placermp
4. Open this directory in terminal and run:
> git clone https://github.com/chefr/placermp
5. From the directory you cloned placermp into, run:
> python setup.py install .

An example of using the package can be found in placermp/examples/revda_placer.py.

# Input data format
An example of the format of the input data is represented in the files placermp/examples/revda_ref_data.txt and placermp/examples/revda_test_data.txt. Currently, the reference (training) and test data should be in the form of ordinary text files, which contain some model data (after #PARAMETERS title) as well as a list of independent parameters in the form of separate sections, separated by a # sign, followed by a section title and a matrix of corresponding values.

Required model parameters: ROWS, COLUMNS, START_ROW, END_ROW, START_COLUMN, END_COLUMN. Their meaning is clear from the name. Row and column counting starts at 1. Both start and end values are included in the model.

Data can be given in quantitative or nominal scales. In the case of a nominal scale "_NOMINAL" must be added after the section title. If the section contains data that will be forecasted (for example in a training model) you must specify '*' after the title.

It is not a problem if some values are missing from the data as long as they are not in the simulation interval. But in place of gaps there should be an arbitrary character (for example "-").

It is permissible to specify as many additional sections as you like for each data section on separate lines after the title in the form +ADDITIONAL_TITLE f: FORMULA. FORMULA is a valid expression in Python, consisting of mathematical operations +, -, *, /, %, **, (, ), as well as calling some functions, such as max(), min(), abs(). The base values are taken from the matrix of section values below. References to the cells of the array are specified in the form [row, column].

It is also possible to specify similar formulas in the PARAMETERS section in the form FORMULA_TITLE = f: FORMULA, which makes it possible to use this formula as a function by its name below.

Please see the comments in the program code for details.