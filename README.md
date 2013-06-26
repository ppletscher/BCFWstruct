#BCFWstruct

BCFWstruct is a Matlab implementation of the Block-Coordinate Frank-Wolfe solver
for Structural SVMs. For more information about the algorithm please check our
[ICML 2013 paper](http://jmlr.org/proceedings/papers/v28/lacoste-julien13-supp.pdf).

The code is organized as follows:
* `solvers` contains the optimization methods, including the block-coordinate 
  Frank-Wolfe solver (BCFW). If you want to use BCFW in your project, you most
  likely only need to run an `addpath(genpath('solvers'))` in your Matlab sources.
* `demos` contains the application-dependent code, such as MAP decoding or the
  feature map computation. The source code includes a sequence prediction demo
  for optical character recognition (OCR).
* `data` is initally empty, it is used to store the data files required for the
  demos.


##Getting Started

1. You need a working installation of Matlab.
2. Clone the git repository: `git clone git@github.com:ppletscher/BCFWstruct.git` (or if you don't want to use git, you can download a zip file of the latest version of the code by following [this link](https://github.com/ppletscher/BCFWstruct/archive/master.zip)).
3. Obtain the data files required to run the demos. On Unix systems you can
   simply run `./fetch_data.sh`. On Windows, you can use
   [Cygwin](http://www.cygwin.com/) or manually download the listed files and
   put them in the data folder.
4. For the OCR demo change to `demo/chain` and run `ocr` from within Matlab.


##Usage

If you would like to use the BCFW solver for your own structured output
prediction problem, you will need to implement three functions:

* The feature map.
* The maximization oracle.
* The loss function.

You can find an example implementation in the `demo/chain` folder. For an
overview of the exact usage and the supported options, please check the Matlab
documentation of the solvers.

Note that BCFW uses a similar calling interface as the one from the Matlab
wrapper to SVM^struct [implemented by Andrea Vedaldi](http://www.vlfeat.org/~vedaldi/code/svm-struct-matlab.html). Users of SVM^struct can thus easily use BCFW with only a tiny change in their code (see the Matlab documentation of solverBCFW for more details).


##Citation

Please use the following BibTeX entry to cite this software in your work:

    @inproceedings{LacosteJulien2013,
      author    = {Lacoste-Julien, Simon and Jaggi, Martin and Schmidt, Mark and Pletscher, Patrick},
      title     = {Block-Coordinate {F}rank-{W}olfe Optimization for Structural {SVMs}},
      booktitle = {ICML},
      year      = {2013},
    }


##Authors

* [Simon Lacoste-Julien](http://www.di.ens.fr/~slacoste/)
* [Martin Jaggi](http://www.cmap.polytechnique.fr/~jaggi/)
* [Mark Schmidt](http://www.di.ens.fr/~mschmidt/)
* [Patrick Pletscher](http://pletscher.org)


##Octave Support

The code also works with [Octave](http://www.octave.org), this was tested with Octave 3.6.4. In order to get the progress update working, we recommend running `more off` before calling our structured SVM solvers.
