# src/app/services/hipergator_bundle.py
import io
import tarfile
import time
from typing import Any, Optional, Tuple


def generate_mail_block(email: Optional[str]) -> str:
    """Return SBATCH mail directives if an email is provided."""
    if not email:
        return ""
    return "#SBATCH --mail-type=ALL\n#SBATCH --mail-user={}".format(email)


def _env_exports(env: Optional[dict]) -> str:
    if not env:
        return ""
    lines = ['export {}="{}"'.format(k, v) for k, v in env.items()]
    return "\n".join(lines)


def make_job_bundle(model_id: str, req: Any) -> Tuple[str, io.BytesIO]:
    """
    Create the working directory name and tar-gzipped bundle containing:
      - train.sbatch
      - run_train.sh
    Returns (workdir, tar_bytes)
    """
    job_slug = "{}-{}".format(model_id, int(time.time()))
    workdir = "~/eigen_jobs/{}".format(job_slug)

    mail_block = generate_mail_block(getattr(req, "email", None))
    modules = getattr(req, "modules", None) or []
    modules_line = "module load {}".format(" ".join(modules)) if modules else ""
    env_block = _env_exports(getattr(req, "env", None))
    run_command = getattr(req, "run_command", "python train.py")

    sbatch_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name={}".format(model_id),
        "#SBATCH --account={}".format(req.account),
        "#SBATCH --qos={}".format(req.qos),
        "#SBATCH --partition={}".format(req.partition),
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task={}".format(req.cpus),
        "#SBATCH --mem={}".format(req.mem),
        "#SBATCH --time={}".format(req.time),
        "#SBATCH --gres=gpu:{}:{}".format(req.gpu_type, req.gpus),
    ]
    if mail_block:
        sbatch_lines.append(mail_block)
    sbatch_lines.append("#SBATCH --output {}/slurm_%j.out".format(workdir))
    sbatch_lines.append("")  # blank line before body
    sbatch_body = [
        "pwd; hostname; date",
        modules_line,
        env_block if env_block else "",
        "bash run_train.sh",
        "date",
    ]
    sbatch = "\n".join([line for line in sbatch_lines + sbatch_body if line])

    run_sh = "#!/usr/bin/env bash\nset -euo pipefail\n{}".format(run_command)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in (("train.sbatch", sbatch), ("run_train.sh", run_sh)):
            encoded = data.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(encoded)
            tar.addfile(info, io.BytesIO(encoded))
    buf.seek(0)

    return workdir, buf
