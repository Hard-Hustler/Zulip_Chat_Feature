ingest:
	docker compose up --build ingest

online:
	docker compose up -d --build online

generator:
	docker compose up -d --build generator

batch:
	docker compose --profile batch up --build batch

minio:
	docker compose up -d minio
