# Compute: Brev and the MSOE GB10

| Machine | Use it for | Isaac Sim |
|---|---|---|
| **Brev** (cloud, billed while running) | Isaac data generation, closed-loop sim, big training runs | Yes: Isaac Sim 6.0.1 launchable |
| **MSOE GB10** (DGX Spark, shared, free) | Training, evaluation, VLM experiments | Not yet: needs an aarch64 source build ([below](#isaac-sim-on-the-gb10-unverified)) |
| Your laptop | Code, tests, `make smoke`, CSI deploy | No |

Never put passwords, API keys or tokens in this repo, in commit messages or in chat logs.

## Brev without the launchable (verified 2026-10-19)

Use this when the launchable can't deploy: it is pinned to Brev's AWS account. Verified on a Crusoe `l40s-48gb.1x` (about $1.74/hr, stoppable, Ubuntu 22.04, driver 565):
```bash
brev search --gpu-name L40S --sort price
```
```bash
brev create smartgrasp-l40s --type l40s-48gb.1x
```
```bash
brev refresh && ssh smartgrasp-l40s
```
On the instance:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```bash
uv venv --python 3.12 ~/isaac-venv
```
```bash
uv pip install --python ~/isaac-venv/bin/python "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
```
That is a 24 GB install. Then create `~/isaac/python.sh` so the README's `$ISAAC/python.sh` commands work unchanged:
```sh
#!/bin/sh
export OMNI_KIT_ACCEPT_EULA=YES   # setting this accepts NVIDIA's Omniverse licence
exec "$HOME/isaac-venv/bin/python" "$@"
```
Run `chmod +x ~/isaac/python.sh`, then `export ISAAC=~/isaac`.

The image's driver (565) runs Isaac Sim, but it is too old for the project's CUDA 13 PyTorch: `torch.cuda` reports "driver too old". The provider holds the driver packages, so don't upgrade it. Add the forward-compatibility libraries instead:
```bash
sudo apt-get install -y cuda-compat-13-0
```
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
```
Set that `export` in every shell that trains, or add it to `~/.bashrc`.

## Brev: Isaac Sim 6.0.1 launchable

1. Open the [Isaac launchable](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-35JP2ywERLgqtD0b0MIeK1HnF46) and click **Deploy Launchable**. The first boot takes a while.
   - Pick a GPU with RT cores: L40S or RTX PRO 6000.
   - H100 and A100 cannot render Isaac Sim.
2. On your laptop, install the CLI and log in once:
   ```bash
   brew install brevdev/homebrew-brev/brev
   ```
   ```bash
   brev login
   ```
3. Sync the new instance into your SSH config, then connect:
   ```bash
   brev refresh
   ```
   ```bash
   brev ls
   ```
   ```bash
   ssh <instance-name>
   ```
4. Copy the repo up. The repo is private, so push from your laptop rather than storing GitHub credentials on a shared instance. From the repo root on your laptop:
   ```bash
   rsync -az --exclude .venv --exclude .env --exclude data --exclude artifacts --exclude runs ./ <instance-name>:smartgrasp/
   ```
5. On the instance, find the container that has `/isaac-sim` and work inside it. Isaac Sim runs in Docker there. `docker ps` lists the containers; the launchable names one `isaac-lab-vscode`.
6. Run the pipeline. Commands are in the README "Full pipeline" table. Before running Isaac, set:
   ```bash
   export ISAAC=/isaac-sim OMNI_KIT_ACCEPT_EULA=YES
   ```
7. **Stop the instance when you finish.** Brev bills while it runs. Copy results back first, for example `rsync -az <instance-name>:smartgrasp/artifacts/ artifacts/`.

## MSOE GB10 (spark5)

Connect to the MSOE VPN, then:
```bash
ssh hackathon15@spark5.ai.msoe.edu
```
The password is in Riley's access email. Do not write it anywhere in this repo. To stop typing it, install your own key once with `ssh-copy-id hackathon15@spark5.ai.msoe.edu`.

The `hackathon15` account is shared by the whole team, and the machine is shared with other teams:
- **Work in your own folder** (`~/<your-name>/smartgrasp`), so you don't overwrite a teammate's run.
- **Check before a long job.** `who` shows who is logged in, and `nvidia-smi` shows GPU use.
- **Report repeated resource problems** to the address in the access email.

Set up the host pipeline. The GB10 is an ARM (aarch64) machine:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```bash
cd ~/<your-name>/smartgrasp && uv sync && make test
```
```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
If that prints `False`, the installed PyTorch wheel has no CUDA for this machine. Ask the mentor before changing `pyproject.toml`, because that change affects every machine. Training then uses `--device 0`:
```bash
uv run python -m smartgrasp.train data/bev/sim_v1 --name sim_v1 --device 0
```

### Isaac Sim on the GB10 (unverified)

NVIDIA's [DGX Spark playbook](https://build.nvidia.com/spark/isaac/isaac-sim) builds Isaac Sim from source:
- It needs gcc 11 and git-lfs.
- `./build.sh` takes about 11 minutes.
- Afterwards, set `ISAAC=<IsaacSim>/_build/linux-aarch64/release` and `export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"`.

A [forum report](https://forums.developer.nvidia.com/t/isaac-sim-6-0-1-build-failure-on-dgx-spark-kit-file-isaacsim-exp-extscache-0-0-0-failed-to-resolve-dependencies/379069) shows the 6.0.1 build failing on a GB10. Use Brev for Isaac until someone gets this working; when you do, record the exact steps here.
