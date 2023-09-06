{lib, ...}: {
  perSystem = {
    config,
    self',
    inputs',
    pkgs,
    system,
    ...
  }: {
    packages.default = pkgs.callPackage ./package.nix {};

    devShells.default = let
      python = pkgs.python3;
    in
      pkgs.mkShell rec {
        name = "impurePythonEnv";

        # Setup Environment Variables
        LD_LIBRARY_PATH = "/usr/lib/wsl/lib";
        PIP_DISABLE_PIP_VERSION_CHECK = 1;
        LANG = "en_US.UTF-8";
        venvDir = "./.venv";

        # Add dependencies for the environment
        buildInputs = [
          # This executes some shell code to initialize a venv in $venvDir before
          # dropping into the shell
          python.pkgs.venvShellHook

          # Those are dependencies that we would like to use from nixpkgs, which will
          # add them to PYTHONPATH and thus make them accessible from within the venv.
          python.pkgs.black
        ];

        # Include the dependencies from our package to allow us to get a development
        # environment for it
        inputsFrom = [
          self'.packages.default
        ];

        # Run the pip shell hook to give us an editable install of the package
        # and venv shell hook to give us a virtual environment to be used by
        # dev tools and install extra packages.
        shellHook = ''
          runHook preShellHook

          if [ -d "${venvDir}" ]; then
            echo "Skipping venv creation, '${venvDir}' already exists"
            source "${venvDir}/bin/activate"
          else
            echo "Creating new venv environment in path: '${venvDir}'"
            ${python.interpreter} -m venv "${venvDir}"

            source "${venvDir}/bin/activate"
            runHook postVenvCreation
          fi

          if [ -e pyproject.toml ]; then
            tmp_path=$(mktemp -d)
            export PATH="$tmp_path/bin:$PATH"
            export NIX_PYTHONPATH="$tmp_path/${python.sitePackages}"
            mkdir -p "$tmp_path/${python.sitePackages}"
            ${python.interpreter} -m pip install -e . --prefix "$tmp_path" --no-build-isolation >&2
          fi

          runHook postShellHook
        '';

        # Run this command, only after creating the virtual environment
        postVenvCreation = ''
          unset SOURCE_DATE_EPOCH
        '';

        # Now we can execute any commands within the virtual environment.
        # This is optional and can be left out to run pip manually.
        postShellHook = ''
          # allow pip to install wheels
          unset SOURCE_DATE_EPOCH
        '';
      };
  };
}
