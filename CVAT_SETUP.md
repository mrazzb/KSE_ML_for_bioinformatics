# CVAT setup for Practical 7

CVAT is a separate annotation service. Run it with Docker Compose rather than installing its server components into the course `uv` environment.

## Windows: Docker Desktop with WSL2

Open PowerShell as Administrator and install WSL2 if it is not already available:

```powershell
wsl --install
```

Restart Windows if requested. Then:

1. Install Docker Desktop for Windows.
2. Keep the WSL2 backend enabled.
3. In Docker Desktop, open **Settings → Resources → WSL Integration** and enable the Linux distribution you use.
4. Install Git if it is not already available.
5. Use the WSL terminal for the commands below.

Clone the CVAT repository and start the services:

```bash
git clone https://github.com/cvat-ai/cvat.git
cd cvat
docker compose up -d
```

The first start downloads the CVAT, database and queue-service images and may take several minutes. Open CVAT in a supported browser at:

```text
http://localhost:8080
```

Create the first administrator account from the WSL terminal:

```bash
docker exec -it cvat_server bash -ic \
  'python3 ~/manage.py createsuperuser'
```

## Linux

Install Docker Engine and the Docker Compose plugin using the official Docker instructions for your distribution. On Ubuntu, the required packages are Docker Engine, the Docker CLI, containerd, Buildx and the Compose plugin.

Then run:

```bash
git clone https://github.com/cvat-ai/cvat.git
cd cvat
docker compose up -d
docker exec -it cvat_server bash -ic \
  'python3 ~/manage.py createsuperuser'
```

Open `http://localhost:8080` in the browser.

## Basic annotation workflow

1. Log in and create a new task.
2. Add the image files from `practicals/week7/detection/test` or a private labelled dataset.
3. Define the class label `colony` for the current practical.
4. Annotate bounding boxes using the agreed rules for border objects, touching colonies, artifacts and very small objects.
5. Save and review a pilot subset before annotating the full dataset.
6. Export the task or project in the required annotation format.

For the current notebook, the most convenient format is **YOLO 1.1**. Its export contains Darknet-style files such as `obj.names`, `train.txt`, `obj_train_data/` and one `.txt` file per annotated image. The notebook converts these pairs into the Ultralytics `images/train`, `images/val`, `labels/train` and `labels/val` structure before training.

Place the extracted private export locally under:

```text
practicals/week7/detection/colonies_annotation/
```

Do not commit the full annotation export or trained model weights. The public repository contains only one example image/label pair so students can inspect the format.

## Useful Docker commands

Run these commands from the cloned `cvat` directory:

```bash
docker compose ps                 # service status
docker compose logs -f cvat_server # live server logs
docker compose stop               # stop without removing data
docker compose start              # start again
docker compose down               # stop and remove containers; volumes remain unless explicitly removed
```

Keep CVAT bound to localhost for local teaching. Do not expose port 8080 directly to the public internet without authentication, HTTPS and a deliberate backup policy.

## Official references

- [CVAT installation guide](https://docs.cvat.ai/docs/administration/basics/installation/)
- [CVAT YOLO format](https://docs.cvat.ai/docs/manual/advanced/formats/format-yolo/)
- [CVAT dataset formats](https://docs.cvat.ai/docs/dataset_management/formats/)
