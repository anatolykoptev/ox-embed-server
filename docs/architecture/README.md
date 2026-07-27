# embed-server — архитектурные диаграммы

LikeC4 source. Build → static SPA → `https://m.krolik.run/c4/embed-server/`.

Single process serving three model classes concurrently:

## Файлы
- `catalog-info.yaml` — Backstage descriptor
- `embed-server.c4` — LikeC4 source (skeleton — расширь)

## Edit + publish
```bash
$EDITOR embed-server.c4
~/bin/c4-publish embed-server
```

**TODO**: обогатить containers / components / dynamic views.
