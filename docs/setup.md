# Local setup

Install and start Docker Desktop on macOS, Linux, or Windows. On Windows, use Linux containers. A Linux Docker Engine installation with the Compose plugin also works. From the repository root run:

```bash
docker compose up -d --build --wait
```

The first start needs internet access for container images, the locked Python packages, and the roughly 523 MB `qwen3:0.6b` model. The running services and inference are local. Subsequent starts reuse cached layers and model weights. [Ollama model details](https://ollama.com/library/qwen3:0.6b)

| Service | Inspect or connect |
| --- | --- |
| Mailpit | Open [localhost:8025](http://localhost:8025) and look for **Email triage startup check** addressed to `complaints@demo.test`. Use `127.0.0.1:1025` for host SMTP or `mailpit:1025` inside Compose. |
| Ollama | Open [localhost:11434/api/tags](http://localhost:11434/api/tags) to see the installed model. Containers use `http://ollama:11434`. |
| MLflow | Open [localhost:5000](http://localhost:5000), choose **email-triage-starter**, then **Traces**. Containers use `http://mlflow:5000`. |
| Python app | Invoke the starter with the command below or open a shell with `docker compose exec app sh`. The image already has Python 3.12, uv, and the locked dependencies. |

## Run the graph

```bash
docker compose exec app python -m email_triage.agent "Say hello in one short sentence."
```

The CLI prints the model reply and the MLflow trace ID. Each startup verification sends one synthetic message and records one starter trace. Stop the stack with `docker compose down` to keep the data volumes for the next start.

## Troubleshooting

- For a startup failure, run `docker compose ps -a` and `docker compose logs verify`. If the verifier never started, inspect `docker compose logs mailpit ollama mlflow` for an unhealthy service. An exited verifier with code 0 is expected after success.
- If a host port is already in use, stop the process using port 1025, 5000, 8025, or 11434 and rerun the setup command.
- If the first model pull fails, check network access for Docker and run the setup command again. The verifier skips the pull when the model is already installed.
- Docker Desktop on macOS runs containerized Ollama on the CPU because it cannot pass the Apple GPU through to the container. Native Ollama with Metal is an optional future alternative for benchmarking. [Ollama FAQ](https://github.com/ollama/ollama/blob/main/docs/faq.mdx#how-do-i-use-ollama-with-gpu-acceleration-in-docker)
- To rerun the verification and see its output, use `docker compose run --rm verify`. This creates a fresh sample email and trace.

## Publish the documentation

A repository maintainer should open **Settings → Pages** and choose **GitHub Actions** under **Build and deployment → Source**. The **Docs** workflow builds MkDocs with Material and deploys on pushes to `main`. You can also run it manually on `main` from the Actions tab. [GitHub Pages setup](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

To preview the Markdown locally if you have uv installed:

```bash
uv run --locked --only-group docs mkdocs serve
```
