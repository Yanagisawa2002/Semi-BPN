# GitHub Sync

The project root is now a local Git repository. Large data and generated results
are ignored by Git and should be copied to the cloud separately.

## First Push

Create an empty GitHub repository, then run from the project root:

```bash
git remote add origin <YOUR_GITHUB_REPO_URL>
git branch -M main
git push -u origin main
```

If you use HTTPS, GitHub may ask for a personal access token. If you use SSH,
make sure the machine has an SSH key added to GitHub.

## Cloud Clone

On the cloud server:

```bash
git clone --recursive <YOUR_GITHUB_REPO_URL> BPNVer
cd BPNVer
git submodule update --init --recursive
```

Then copy the local-only data bundle to:

```text
BPNVer/data/cloud_run/
```

## Pulling Later Updates

On the cloud server:

```bash
cd BPNVer
git pull --recurse-submodules
git submodule update --init --recursive
```

The data bundle is ignored by Git, so pulling code updates will not overwrite
`data/cloud_run/`.

## Current Submodule

The original BioPathNet code is tracked as a submodule:

```text
biopathnet/original -> https://github.com/emyyue/BioPathNet.git
```

This keeps the original implementation stable while the project code lives under
`mechrep/`.
