{ pkgs ? import <nixpkgs> {} }:

with pkgs;

mkShell {

  # Add our own BayesPy package to Python path
  PYTHONPATH = toString ./.;

  buildInputs = [
    (
      python3.withPackages (
        ps: with ps; [

          # Core
          numpy
          scipy
          h5py
          matplotlib

          # Dev
          ipython
          nose

          # Docs
          sphinx
          sphinxcontrib-tikz
          sphinxcontrib-bayesnet
          sphinxcontrib-bibtex
          sphinxcontrib-katex
          nbsphinx
        ]
      )
    )
  ];
}
