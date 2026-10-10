{
  description = "agent-rl-projects — companion code for the Agent RL handbook";

  inputs = {
    # The same channel and the same locked revision as the handbook repository
    # (nixpkgs 7a0f122f5090), so both shells resolve to identical store paths on the
    # reference machine instead of downloading two copies of Python.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));

      # The two libraries datasets/pull.py needs, pinned here rather than installed by pip:
      # what a pull records should not depend on when somebody last ran an installer. They
      # are imported inside the functions that use them, so nothing else can come to depend
      # on them by accident - and the gate below runs without huggingface_hub at all.
      pullerPython = pkgs: pkgs.python3.withPackages (ps: [ ps.huggingface-hub ps.pyyaml ]);
    in
    {
      formatter = forAllSystems (pkgs: pkgs.nixfmt-rfc-style);

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          # Deliberately small. The contract layer's only dependency is the standard
          # library, so the shell that runs its tests needs an interpreter and nothing that
          # carries a GPU with it - and `scripts/check_repo.py --imports` keeps it that way.
          packages = [
            (pullerPython pkgs)
            pkgs.git # the gate reads the git index; see AGENTS.md
            pkgs.coreutils # timeout, which macOS's own shell does not ship
            pkgs.ripgrep
            pkgs.fd
            pkgs.jq
          ];

          shellHook = ''
            echo "agent-rl-projects dev shell — python $(python3 -V | sed 's/^Python //') with huggingface-hub ($(uname -m) $(uname -s))"
            echo "  bash scripts/verify.sh           # the gate: tests, fixtures, characters/payloads/secrets"
            echo "  nix build --no-link              # the same gate in a sandbox; no ./result link left behind"
            echo "  python3 datasets/pull.py --list  # datasets, and what a pull would record"
          '';
        };
      });

      # The gate as a build: `nix build` fails when any check fails, and it cannot see the
      # network. The run record goes to $TMPDIR because the store is read-only - the same
      # constraint a CI job has, which is why the check script takes ARL_RUNS_DIR.
      #
      # Note what is *not* here: huggingface_hub. The gate is offline by construction, so the
      # pull tool's network library has no business in it - pyyaml is present only so the
      # manifest's own contract tests can run.
      packages = forAllSystems (pkgs: {
        default = pkgs.stdenvNoCC.mkDerivation {
          pname = "agent-rl-projects-gate";
          version = "unstable";
          src = ./.;
          nativeBuildInputs = [
            (pkgs.python3.withPackages (ps: [ ps.pyyaml ]))
            pkgs.git
            pkgs.coreutils
          ];
          dontConfigure = true;
          buildPhase = ''
            runHook preBuild
            export HOME="$TMPDIR"
            export ARL_RUNS_DIR="$TMPDIR/runs"
            bash scripts/verify.sh
            runHook postBuild
          '';
          installPhase = ''
            mkdir -p "$out"
            echo "the gate passed on $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$out/STAMP"
          '';
        };
      });
    };
}
