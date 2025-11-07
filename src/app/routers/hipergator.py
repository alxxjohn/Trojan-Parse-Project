import paramiko
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.hipergator_bundle import make_job_bundle

router = APIRouter(prefix="/api", tags=["HiperGator"])


class HpgSubmit(BaseModel):
    account: str
    qos: str
    partition: str
    time: str
    gpus: int
    gpu_type: str = "a100"
    cpus: int
    mem: str
    email: str | None = None
    modules: list[str] = ["cuda/12.2", "python/3.10"]
    env: dict[str, str] = {}
    run_command: str = "python train.py"
    notes: str | None = None


def ssh_client(host, user, pkey_path):
    key = paramiko.RSAKey.from_private_key_file(pkey_path)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(hostname=host, username=user, pkey=key)
    return c


@router.post("/{model_id}/hipergator/submit")
def send_to_hipergator(model_id: str, req: HpgSubmit):
    # 1) build bundle
    workdir, bundle = make_job_bundle(model_id, req)

    # 2) SSH/SCP and submit ...
    cli = ssh_client(
        os.environ["HPG_HOST"], os.environ["HPG_USER"], os.environ["HPG_PKEY"]
    )
    sftp = cli.open_sftp()
    cli.exec_command(f"mkdir -p {workdir}")
    remote_tar = f"{workdir}/bundle.tar.gz"
    with sftp.open(remote_tar, "wb") as f:
        f.write(bundle.read())
    cli.exec_command(f"cd {workdir} && tar xzf bundle.tar.gz && chmod +x run_train.sh")
    # ... rest of sbatch + DB persistence ...
