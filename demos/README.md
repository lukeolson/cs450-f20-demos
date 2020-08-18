# Jupyter notebook demo tools

`prepare-ipynb` is a small tool that allows you to automate modifications to
Jupyter notebooks that make them suitable for in-class demos.

In particular, you can:

* `clear-marked-inputs`: Clears input cells marked with `#clear`, allowing
  you to select pieces of code to be interactively re-developed in class.

* `clear-output`: Remove all output cells from the notebook, so that the
  results of the computation are more of a 'surprise' in class.

* `remove-marks`: Removes `#clear` marks to produce material suitable for
  distribution to students.

**Note:** Despite the name, the current version also supports Jupyter (v4)
notebooks.
