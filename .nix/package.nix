{python3}: let
  inherit
    (python3.pkgs)
    buildPythonApplication
    pythonOlder
    pytorch-bin
    torchvision-bin
    ;
in
  buildPythonApplication rec {
    pname = "blue-team-nd";
    version = "0.0.1";
    format = "pyproject";
    disabled = pythonOlder "3.8";

    HOME = ".";

    src = ../.;

    propagatedBuildInputs = [
      pytorch-bin
      torchvision-bin
    ];
  }
