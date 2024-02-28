{ pkgs ? import <nixpkgs> {} }:

with pkgs;

mkShell {

  # Add our own BayesPy package to Python path
  PYTHONPATH = toString ./.;

  buildInputs = [

    pandoc
    # Full LaTeX needed for anyfontsize.sty
    # Medium LaTeX would be sufficient for dvisvgm
    texlive.combined.scheme-full

    (
      python3.withPackages (
        ps: with ps; let
          truncnorm = buildPythonPackage rec {
            pname = "truncnorm";
            version = "0.0.2";
            src = fetchPypi {
              pname = pname;
              version = version;
              sha256 = "sha256-D6spzLLfdmUaE2+J+A6Va7CZzeb9prJr3nb5SNAWmOg=";
            };
            doCheck = false;
            buildInputs = [ setuptools_scm ];
            propagatedBuildInputs = [ scipy numpy ];
          };
        in [

          # Core
          numpy
          scipy
          h5py
          matplotlib
          truncnorm

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
