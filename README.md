# RAG Server (FastAPI + Qdrant + LLM)

Sistem za delo s tvojimi dokumenti (Retrieval-Augmented Generation). Dokumente razreže na
kratke kose, jih pretvori v vektorje in shrani v Qdrant. Ima dva načina, med katerima
preklopiš v spletnem vmesniku:

- **Vprašanja → odgovori:** postaviš eno ali več vprašanj in dobiš odgovore z viri iz baze.
- **Tekst → vprašanja:** prilepiš ali odpreš tekst in dobiš odprta vprašanja z odgovori.

Trenutno je baza napolnjena s čebelarskim gradivom iz `dataset/data`.

**Embedder in LLM lahko tečeta na tem računalniku (v Dockerju) ali pa ju uporabljaš preko
URL-ja** (drug strežnik, API). Izbereš v `.env`.

## Vsebina

1. [Zahteve](#zahteve)
2. [Namestitev](#namestitev)
3. [Uporaba](#uporaba)
4. [Skripte](#skripte)
5. [Izbira LLM-ja in embedderja](#izbira-llm-ja-in-embedderja)
6. [Iskanje: hibridno, reranker, velikost kosov](#iskanje-hibridno-reranker-velikost-kosov)
7. [Brisanje podatkov](#brisanje-podatkov-začetek-na-čisto) · [Restart](#restart) · [Dodajanje dokumentov](#dodajanje-dokumentov)
8. [Evalvacija](#evalvacija)
9. [Postavitev na strežnik](#postavitev-na-strežnik) · [Težave](#težave)

## Zahteve

| Kaj | Zakaj |
|---|---|
| **Docker** z Docker Compose | Poganja vse tri kontejnerje. Mac/Windows: [Docker Desktop](https://www.docker.com/products/docker-desktop/). Linux: [Docker Engine](https://docs.docker.com/engine/install/) + compose plugin |
| **Git** | Za prenos projekta |
| **Bash** | Za skripte v `scripts/`. Na Macu in Linuxu je že nameščen, na Windowsih uporabi [WSL](https://learn.microsoft.com/windows/wsl/install) |
| **~10 GB prostora na disku** | Docker image-i, modela `qwen3.5:4b` (~3,4 GB) in `bge-m3` (~1,2 GB) ter baza. Z `LOCAL_MODELS=true` še ~5 GB |
| **8 GB RAM ali več** | Za modela na CPU-ju. Če modela tečeta drugje (URL), zadošča ~1 GB |
| NVIDIA GPU (neobvezno) | Z ≥6 GB VRAM je odgovor gotov v nekaj sekundah, na CPU-ju v 20–40 s. Potrebuješ gonilnik in [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |

Python ni potreben, razen če aplikacijo poganjaš brez Dockerja (`scripts/start-local.sh`,
Python 3.11 ali 3.12).

Preveri, da je Docker nameščen in zagnan:

```bash
docker --version
docker compose version     # ali: docker-compose version
docker info                # mora delovati brez napake
```

## Namestitev

### 1. Prenesi projekt

```bash
git clone https://github.com/FlorijanPeric/RAG_Server.git
cd RAG_Server
```

### 2. Nastavi `.env` (neobvezno)

```bash
cp .env.example .env
```

Privzete nastavitve delujejo brez sprememb: embedder `bge-m3` in LLM `qwen3.5:4b` v
kontejnerju `models`, hibridno iskanje, kosi po 300 znakov. Če `.env` ne ustvariš, ga
`scripts/start.sh` ustvari sam.

**Z NVIDIA GPU-jem** priporočamo še reranker, ki opazno izboljša iskanje (glej
[Iskanje](#iskanje-hibridno-reranker-velikost-kosov)). V `.env` nastavi:

```bash
RERANK_MODE=api
```

Če želiš drug LLM ali embedder (npr. preko URL-ja), uredi `.env` pred zagonom. Glej
[Izbira LLM-ja in embedderja](#izbira-llm-ja-in-embedderja).

> **Mac:** Docker na Macu ne more uporabljati GPU-ja, zato je model v kontejnerju počasen.
> Hitreje je, če [Ollamo namestiš neposredno](https://ollama.com/download), zaženeš
> `ollama pull qwen3.5:4b` in `ollama pull bge-m3` ter v `.env` nastaviš
> `OLLAMA_URL=http://host.docker.internal:11434`.

### 3. Zaženi

```bash
scripts/start.sh
```

Zažene tri kontejnerje:

| Kontejner | Vloga |
|---|---|
| `models` | Ollama z embedderjem (`bge-m3`) in LLM-jem (`qwen3.5:4b`), po želji še reranker. Uporabi NVIDIA GPU, če ga Docker ima |
| `web` | Spletni vmesnik in API (FastAPI) |
| `qdrant` | Vektorska baza |

Prvi zagon traja nekaj minut (5–20, odvisno od povezave), ker:
1. zgradi Docker image-e,
2. prenese modela v kontejner `models`,
3. v ozadju naloži dokumente iz `dataset/data` v bazo.

Naslednji zagoni trajajo nekaj sekund. Skripta na koncu izpiše, kje teče kaj, npr.:

```
LLM:      qwen3.5:4b in the models container
Embedder: bge-m3 in the models container
Search:   hybrid, reranker: off
GPU:      NVIDIA GPU for the models container
```

### 4. Preveri, da deluje

```bash
curl http://localhost:8000/health        # → {"status":"ok"}
scripts/init.sh --check                  # preveri embedder, Qdrant, iskanje in LLM
```

`init.sh --check` izpiše nekaj takega:

```
embedder: api:bge-m3 @ http://models:11434/api/embed
  ok, dimension=1024
qdrant:   http://qdrant:6333 (collection 'Beees')
  ok, has_data=True
retrieval: hybrid (candidates=20, chunk=300 chars)
llm:      ollama:qwen3.5:4b @ http://models:11434
  ok, replied: 'OK'
```

Če je `has_data=False`, se dokumenti še nalagajo. Napredek spremljaš z:

```bash
docker compose logs -f web | grep auto-ingest
```

Ko se izpiše `[auto-ingest] done`, je sistem pripravljen.

> Če imaš samo samostojni `docker-compose` (npr. Homebrew na Macu), uporabi
> `docker-compose` namesto `docker compose`. Skripte v `scripts/` znajo uporabiti oba.

## Uporaba

**V brskalniku:** odpri **http://localhost:8000**. Zgoraj izbereš način:

| Način | Kaj vpišeš | Kaj dobiš |
|---|---|---|
| **Vprašanja → odgovori** | Eno vprašanje ali več vprašanj, vsako v svoji vrstici. `Enter` pošlje | Odgovor na vsako vprašanje z viri iz baze |
| **Tekst → vprašanja** | Tekst (prilepiš ali z gumbom **Odpri datoteko** prebereš `.pdf`, `.txt`, `.docx`, `.xls`). `Ctrl+Enter` / `Cmd+Enter` pošlje | Odprta vprašanja s kratkimi odgovori. Gumb **Kopiraj** jih skopira |

Nastavitve v vmesniku:
- **Top-k:** koliko kosov besedila iz baze dobi LLM (privzeto `TOP_K` iz `.env`).
- **Uporabi LLM** (vprašanja): izklopljeno vrne samo najdene odlomke, brez LLM-ja.
- **Št. vprašanj** in **Uporabi bazo** (tekst → vprašanja): koliko vprašanj naj nastane in ali
  naj LLM za odgovore uporabi tudi bazo, ne samo podanega teksta.
- **Dodaj v bazo:** naloži odprto datoteko v bazo.

**V terminalu:** `scripts/chat.sh`. Glej [Uporaba brez spletnega vmesnika](#uporaba-brez-spletnega-vmesnika).

**Ustavitev:** `scripts/stop.sh`. Podatki in modeli ostanejo shranjeni; ob naslednjem
`scripts/start.sh` so takoj na voljo.

## Skripte

| Ukaz | Kaj naredi |
|---|---|
| `scripts/start.sh` | Zgradi in zažene kontejnerje (`--logs` za sprotni izpis logov) |
| `scripts/start-local.sh` | Zažene aplikacijo neposredno na računalniku, brez Dockerja za aplikacijo. Qdrant zažene v Dockerju ali, če Dockerja ni, kot samostojen program. `--background` = teče naprej po odjavi |
| `scripts/stop.sh` | Ustavi vse. Podatki ostanejo shranjeni (`--volumes` izbriše tudi bazo in modele) |
| `scripts/init.sh` | Preveri embedder, Qdrant, iskanje in LLM ter naloži `dataset/data`, če je baza prazna |
| `scripts/init.sh --reset` | Izbriše bazo in vse naloži znova |
| `scripts/init.sh --check` | Samo preveri, ali vse deluje |
| `scripts/init.sh --force` | Naloži dokumente, čeprav baza ni prazna (lahko nastanejo duplikati) |
| `scripts/init.sh --dir moji_dokumenti` | Naloži drugo mapo |
| `scripts/chat.sh` | Klepet v terminalu, s polnimi izpisi napak |

`init.sh` in `chat.sh` tečeta v kontejnerju `web`, če je zagnan, sicer lokalno v `.venv`.

## Uporaba brez spletnega vmesnika

Spletni vmesnik ni obvezen. Ko je sistem zagnan (`scripts/start.sh`), ga lahko uporabljaš
samo iz terminala.

**Klepet v terminalu:**

```bash
scripts/chat.sh
```

```
You: Kako čebele sporočajo lokacijo hrane?

[llm] Čebele uporabljajo ples ...
  4 source chunk(s):
    [1] (dataset/data/cebele.pdf, score=0.712) ...
```

Ukaz `/llm` vklopi ali izklopi LLM, `exit` konča klepet.

**API s `curl`** (za skripte in druge programe):

```bash
# vprašanje → odgovor
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Kako čebele sporočajo lokacijo hrane?"}'

# tekst → vprašanja z odgovori
curl -X POST http://localhost:8000/questions \
  -H "Content-Type: application/json" \
  -d '{"text": "Matica je edina spolno razvita samica v panju ...", "num_questions": 3}'
```

Vsi endpointi so opisani v razdelku [API](#api).

## Dostop z drugih naprav

Privzeto (`BIND_HOST=0.0.0.0`) sistem posluša na vseh omrežnih vmesnikih, ne samo na `localhost`.

- **Isto omrežje (WiFi/LAN):** poišči IP računalnika (`ipconfig getifaddr en0` na Macu,
  `hostname -I` na Linuxu) in na drugi napravi odpri `http://<IP>:8000`.
- **Strežnik:** uporabi SSH tunel (glej [Postavitev na strežnik](#postavitev-na-strežnik)).

> **Pozor:** sistem nima prijave. Kdor doseže port, lahko postavlja vprašanja in nalaga
> dokumente. Na deljenem strežniku nastavi `BIND_HOST=127.0.0.1` in uporabljaj SSH tunel.
> Porta nikoli ne odpri javno na internetu.

## Izbira LLM-ja in embedderja

Nastaviš v `.env`. Po spremembi zaženi `scripts/stop.sh` in `scripts/start.sh`.

Kontejner `models` se zažene samo, če ga embedder ali LLM potrebuje (`OLLAMA_URL` je prazen).
Če oba uporabljaš preko URL-ja, tečeta samo `web` in `qdrant`.

### LLM (odgovori in generiranje vprašanj)

| `LLM_MODE` | Ostale nastavitve | Kje teče |
|---|---|---|
| `ollama` (privzeto) | `OLLAMA_URL` prazen, `OLLAMA_MODEL=qwen3.5:4b` | Kontejner `models`, model se prenese samodejno |
| `ollama` | `OLLAMA_URL=http://streznik:11434` | Ollama na drugem strežniku |
| `ollama` | `OLLAMA_URL=http://host.docker.internal:11434` | Ollama, nameščena na tem računalniku |
| `api` | `LLM_API_URL`, `LLM_API_MODEL`, `LLM_API_KEY` | Kateri koli OpenAI-kompatibilen API (vLLM, LM Studio, llama.cpp, OpenAI, hivecore …) |
| `local` | `LLM_LOCAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct`, `LOCAL_MODELS=true` | HuggingFace model, naložen neposredno v aplikacijo (CUDA, Apple MPS ali CPU) |

Če LLM ni dosegljiv, odgovarjanje še vedno deluje. Namesto generiranega odgovora vrne
najbolj ustrezne odlomke (ekstraktivni način). Generiranje vprašanj brez LLM-ja ne deluje.

### Embedder (pretvarja besedilo v vektorje)

| `EMBEDDING_MODE` | Ostale nastavitve | Kje teče |
|---|---|---|
| `api` (privzeto) | `EMBEDDING_API_URL` prazen, `EMBEDDING_API_MODEL=bge-m3` | Na isti Ollami kot LLM (privzeto kontejner `models`) |
| `api` | `EMBEDDING_API_URL`, `EMBEDDING_API_MODEL`, `EMBEDDING_API_KEY` (neobvezen) | Preko URL-ja: OpenAI `/v1/embeddings`, Ollama `/api/embed`, hivecore |
| `local` | `EMBEDDING_MODEL=...`, `LOCAL_MODELS=true` | sentence-transformers model v aplikaciji |

> **Pomembno:** po menjavi embedderja stari vektorji niso več združljivi z novim modelom.
> Zaženi `scripts/init.sh --reset`. Če tega ne narediš, strežnik javi napako o neujemanju
> dimenzij.

### `LOCAL_MODELS`

Načini `local` (`EMBEDDING_MODE`, `LLM_MODE` in `RERANK_MODE`) potrebujejo PyTorch v
kontejnerju `web`. Ker je image s PyTorchem nekaj GB večji, ga vklopiš posebej:
`LOCAL_MODELS=true` v `.env`, nato `scripts/start.sh` (image se zgradi znova).

### GPU

`MODELS_GPU=auto` (privzeto) da kontejnerju `models` NVIDIA GPU, če ga Docker ima
(`docker info` vsebuje `nvidia`). `true` ga vsili, `false` izklopi.

## Iskanje: hibridno, reranker, velikost kosov

Kakovost odgovorov je najbolj odvisna od tega, ali iskanje najde pravi kos besedila.
Iskanje poteka v treh korakih:

1. **Hibridno iskanje** (`RETRIEVAL_MODE=hybrid`, privzeto): vektorsko iskanje najde kose s
   podobnim pomenom, BM25 pa kose z enakimi besedami (imena, številke, strokovni izrazi).
   Oba seznama (po `RERANK_CANDIDATES` kosov) se združita z *reciprocal rank fusion*.
   BM25 indeks se zgradi v pomnilniku ob prvem vprašanju in se sam osveži, ko se baza spremeni.
2. **Reranker** (`RERANK_MODE`, privzeto `off`): cross-encoder (`BAAI/bge-reranker-v2-m3`,
   večjezičen) prebere vprašanje skupaj z vsakim kandidatom in jih na novo razvrsti. Je
   natančnejši od vektorjev, a potrebuje GPU: na GPU-ju doda ~0,1–0,3 s na vprašanje, na
   CPU-ju ~5 s.
3. LLM dobi `TOP_K` najboljših kosov.

| Nastavitev | Privzeto | Možnosti |
|---|---|---|
| `RETRIEVAL_MODE` | `hybrid` | `hybrid`, `vector` (samo vektorji), `bm25` (samo ključne besede) |
| `RERANK_MODE` | `off` | `off`; `api` s praznim `RERANK_API_URL` = reranker v kontejnerju `models` na GPU-ju (priporočeno); `api` z `RERANK_API_URL` = kateri koli `/rerank` endpoint (llama.cpp server `--reranking`, Infinity, vLLM, Hugging Face TEI …); `local` = model v kontejnerju `web` na CPU-ju (potrebuje `LOCAL_MODELS=true`) |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Kateri koli cross-encoder s HuggingFace (`local`) ali ime modela za API |
| `RERANK_CANDIDATES` | `20` | Koliko kandidatov pregledata hibridno iskanje in reranker |
| `CHUNK_MAX_CHARS` | `300` | Največja dolžina kosa v znakih. Manjši kosi = natančnejše iskanje, a manj konteksta na kos |
| `CHUNK_OVERLAP_SENTENCES` | `1` | Koliko stavkov se ponovi med sosednjima kosoma |

Primer z rerankerjem na GPU-ju v kontejnerju `models`:

```bash
# .env
RERANK_MODE=api
RERANK_API_URL=
```

`scripts/start.sh` nato image `models` zgradi s PyTorchem (~5 GB več) in ob zagonu prenese
model (~2 GB). Reranker teče na portu 8001 znotraj Docker omrežja; izpis
`[reranker] ready on :8001 (cuda)` v `docker compose logs models` pomeni, da uporablja GPU.

**Vklop in izklop rerankerja:** v `.env` nastavi `RERANK_MODE=api` (vklop) ali
`RERANK_MODE=off` (izklop), nato:

```bash
scripts/stop.sh
scripts/start.sh
```

Baze ni treba znova napolniti, ker reranker na shranjene podatke ne vpliva. Brez rerankerja
`models` ne zaseda dodatnega GPU pomnilnika, image pa se zgradi brez PyTorcha (manjši).
Brez GPU-ja rerankerja raje ne vklopi, ker na CPU-ju doda ~5 s na vprašanje.

Rezultati na čebelarskih vprašanjih (qwen3.5:4b, bge-m3, `TOP_K=4`, glej [Evalvacija](#evalvacija)):

| Iskanje | Kosi | Iskanje najde odgovor | LLM pravilno |
|---|---|---|---|
| samo vektorji | 400 | 17/120 (14 %) | 62 % |
| hibridno | 300 | 20/120 (17 %) | 59 % |
| hibridno + reranker | 400 | 25/120 (21 %) | 62 % |
| hibridno + reranker | 300 | 23/120 (19 %) | **65 %** |

Zgornja meja iskanja je 49/120: pri ostalih vprašanjih odgovora v gradivu ni. Rezultati LLM-ja
med ponovitvami nihajo za nekaj odstotkov.

> Sprememba `CHUNK_MAX_CHARS` velja samo za novo naložene dokumente. Po spremembi zaženi
> `scripts/init.sh --reset`. `RETRIEVAL_MODE` in `RERANK_*` veljata takoj po restartu.

## Brisanje podatkov (začetek na čisto)

Datoteke v `dataset/` se nikoli ne brišejo, briše se samo vektorska baza.

| Želim … | Ukaz |
|---|---|
| izbrisati bazo in znova naložiti `dataset/data` | `scripts/init.sh --reset` |
| popolnoma prazno bazo | v `.env` nastavi `AUTO_INGEST_ENABLED=false`, nato `scripts/init.sh --reset --check` (sicer se ob naslednjem zagonu `dataset/data` naloži znova) |
| izbrisati čisto vse, tudi prenesene modele | `scripts/stop.sh --volumes`, nato `scripts/start.sh` |

## Restart

| Situacija | Ukaz |
|---|---|
| Spremenil si `.env` (npr. vklop/izklop rerankerja) | `scripts/stop.sh` in `scripts/start.sh` |
| Spremenil si kodo | `scripts/start.sh` (znova zgradi image) |
| Nekaj se je zataknilo | `docker compose restart web` (ali `models`, `qdrant`) |
| Lokalni način | `Ctrl+C` (ali `scripts/stop.sh` pri `--background`), nato `scripts/start-local.sh` |

## Dodajanje dokumentov

### Nove datoteke v mapo (priporočeno)

1. Datoteke (`.pdf`, `.txt`, `.docx`, `.xls`; podmape so dovoljene) kopiraj v **`dataset/data/`**.
   Excel `.xlsx` ni podprt, shrani ga kot `.xls`.
2. Zaženi:
   ```bash
   scripts/init.sh --reset
   ```
   To izbriše bazo in znova naloži **celo** mapo `dataset/data/`, skupaj z novimi
   datotekami. Na GPU-ju traja nekaj minut, na CPU-ju dlje. Napredek se izpisuje sproti
   (`ingested ... -> N chunks`), na koncu pa npr. `{'files_found': 9, 'ingested': 9, 'skipped': 0}`.
3. Preveri z `scripts/init.sh --check` (`has_data=True`).

| Ukaz | Kaj naredi | Kdaj |
|---|---|---|
| `scripts/init.sh` | Naloži mapo samo, če je baza prazna | Prvi zagon (to stori že `start.sh` sam) |
| `scripts/init.sh --reset` | Izbriše bazo in naloži vse znova | **Ko dodaš, spremeniš ali izbrišeš datoteke** |
| `scripts/init.sh --force` | Doda vse datoteke še enkrat poleg obstoječih | Skoraj nikoli: ustvari duplikate |
| `scripts/init.sh --reset --dir druga_mapa` | Bazo napolni iz druge mape | Ko želiš drugačno gradivo |

> - Brez `--reset` se ne naloži nič, ker baza že ima podatke.
> - Izbrisana datoteka ostane v bazi, dokler ne zaženeš `--reset`.
> - Vprašanj za evalvacijo ne dajaj v `dataset/data/`: spadajo v `dataset/questions/`,
>   sicer iskanje najde kar vprašanja in rezultati evalvacije niso več pravi.
> - Po spremembi `CHUNK_MAX_CHARS` ali embedderja je `--reset` obvezen.

**Na strežniku:** datoteke prenesi v mapo projekta in tam zaženi reset:

```bash
scp moj_dokument.pdf streznik:~/RAG_Server_deploy/dataset/data/
ssh streznik 'cd ~/RAG_Server_deploy && scripts/init.sh --reset'
```

### Posamezna datoteka (brez reseta)

- **Spletni vmesnik:** gumb **Odpri datoteko**, nato **Dodaj v bazo**. Datoteka se doda k
  obstoječim podatkom. Pri naslednjem `--reset` je ne bo več, razen če jo daš tudi v `dataset/data/`.
- **API:**
  ```bash
  curl -X POST http://localhost:8000/ingest/pdf -F "file=@dokument.pdf"

  curl -X POST http://localhost:8000/ingest/text \
    -H "Content-Type: application/json" \
    -d '{"text": "Čebele komunicirajo s plesom.", "source": "zapiski"}'
  ```

## Evalvacija

`eval_rag.py` izmeri kakovost na čebelarskih vprašanjih z več možnimi odgovori iz
`dataset/questions` (pravilen odgovor je v `.docx` označen z zeleno). Uporablja enake
nastavitve kot strežnik, zato lahko primerjaš embedderje, LLM-je in načine iskanja.

```bash
# v kontejnerju web (ali lokalno: .venv/bin/python eval_rag.py ...)
docker compose exec web python eval_rag.py                  # samo iskanje
docker compose exec web python eval_rag.py --llm 129        # + LLM odgovori na vsa vprašanja
docker compose exec web python eval_rag.py --llm 129 --no-context   # LLM brez baze (primerjava)
docker compose exec -e RETRIEVAL_MODE=vector web python eval_rag.py # drug način iskanja
```

Izpiše:
- **retrieval:** pri koliko vprašanjih je besedilo pravilnega odgovora med najdenimi kosi, in
  zgornjo mejo (pri koliko vprašanjih se odgovor sploh pojavi v bazi),
- **llm:** koliko odgovorov a/b/c je LLM izbral pravilno (naključno ugibanje ≈ 33 %).

LLM brez baze doseže 41 %, z bazo do 65 % (rezultati po nastavitvah so v razdelku
[Iskanje](#iskanje-hibridno-reranker-velikost-kosov)).

## Arhitektura

```
                     ┌──────────────────────────────┐
 brskalnik ─▶ :8000  │ web (FastAPI + spletni UI)   │
 chat.sh / curl      │ hibridno iskanje, BM25,      │
                     │ (reranker), generiranje      │
                     └───────┬──────────────┬───────┘
                             │              │
                    ┌────────▼───┐   ┌──────▼─────────────────────────┐
                    │ qdrant     │   │ models (GPU): Ollama :11434    │
                    │ :6333      │   │  bge-m3 + qwen3.5:4b           │
                    └────────────┘   │ + reranker :8001 (neobvezno)   │
                                     │ ali: modeli preko URL          │
                                     └────────────────────────────────┘
```

| Datoteka | Vloga |
|---|---|
| `app.py` | FastAPI strežnik, API, samodejno nalaganje ob prvem zagonu |
| `rag_pipeline.py` | Jedro: embedding, Qdrant, hibridno iskanje, reranker, odgovori, generiranje vprašanj (vsi načini LLM-ja) |
| `bm25.py` | BM25 iskanje po ključnih besedah |
| `splitter.py` | Rezanje besedila na kose |
| `ingest_dataset.py` | Branje `.pdf`, `.txt`, `.docx`, `.xls` |
| `init_rag.py` | Inicializacija in ponastavitev (kliče ga `scripts/init.sh`) |
| `eval_rag.py` | Evalvacija na čebelarskih vprašanjih |
| `chat_cli.py` | Klepet v terminalu |
| `static/index.html` | Spletni vmesnik |
| `models/` | Docker image za kontejner `models` (Ollama, ki ob zagonu prenese modele; `reranker_server.py` = reranker na GPU-ju) |
| `docker-compose.yml`, `docker-compose.gpu.yml` | Kontejnerji; GPU datoteko doda `scripts/start.sh` samodejno |

## API

| Endpoint | Namen |
|---|---|
| `GET /` | Spletni vmesnik |
| `GET /health` | Preverjanje delovanja |
| `POST /ask` | `{question, top_k, use_llm}` → `{answer, contexts, mode}`. `top_k` je neobvezen (privzeto `TOP_K`) |
| `POST /questions` | `{text, num_questions, top_k, use_context}` → `{questions: [{question, answer}], contexts, text_truncated}`. `num_questions` 1–20 (privzeto 5). Tekst daljši od 6000 znakov se skrajša |
| `POST /extract` | Datoteka (`.pdf`, `.txt`, `.docx`, `.xls`) → `{filename, text}`. Ne naloži v bazo |
| `GET /config` | Nastavitve za vmesnik (`top_k`, `retrieval`) |
| `POST /ingest/pdf` | Nalaganje PDF-ja |
| `POST /ingest/text` | Nalaganje besedila (`text`, `source`) |
| `POST /ingest/file` | Nalaganje UTF-8 datoteke po poti na strežniku |
| `POST /debug/embed` | Vrne vektorje za podana besedila (za preverjanje embedderja) |

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Kako čebele sporočajo lokacijo hrane?", "top_k": 4, "use_llm": true}'
```

`mode` v odgovoru je `"llm"` (generiran odgovor) ali `"extractive"` (samo odlomki, če je
`use_llm: false` ali LLM ni dosegljiv; razlog je v `llm_error`).

## Postavitev na strežnik

### 1. Prenos

```bash
./package.sh        # počisti kopijo v ../RAG_Server_deploy
rsync -az --delete --exclude .env --exclude .venv --exclude .qdrant --exclude logs \
  ../RAG_Server_deploy/ streznik:~/RAG_Server_deploy/
```

> `--exclude` je pomemben: brez njega `--delete` na strežniku izbriše `.env`, Python okolje,
> bazo in loge. Po spremembi kode ponovi oba ukaza.

### 2. Zagon z Dockerjem (priporočeno)

Uporabnik mora biti v skupini `docker`. Za GPU morata biti nameščena NVIDIA gonilnik in
NVIDIA Container Toolkit (`docker info | grep -i nvidia` mora nekaj izpisati).

```bash
ssh streznik
cd ~/RAG_Server_deploy
cp .env.example .env     # uredi .env, glej primer spodaj
scripts/start.sh
```

Primer `.env` za deljen strežnik (vse samo na `127.0.0.1`, prosti porti):

```bash
BIND_HOST=127.0.0.1
RAG_API_HOST_PORT=8010
QDRANT_HOST_PORT=6340
OLLAMA_HOST_PORT=11460
MODELS_GPU=auto
RERANK_MODE=api          # reranker na GPU-ju v kontejnerju models
```

Porte prilagodi tistim, ki so na strežniku prosti (`ss -tln`). Po posodobitvi kode znova
zaženi `scripts/start.sh`; če si spremenil `CHUNK_MAX_CHARS`, še `scripts/init.sh --reset`.

### Zagon brez Dockerja za aplikacijo

Npr. ko na strežniku že teče Ollama, ki posluša samo na `127.0.0.1`:

```bash
scripts/start-local.sh --background
```

`--background` pomeni, da sistem teče naprej tudi po odjavi. Logi so v `logs/rag-api.log`,
ustaviš pa ga s `scripts/stop.sh`. Skripta sama:
- ustvari Python okolje, tudi če na strežniku manjka paket `python3-venv`,
- namesti PyTorch samo, če ga potrebuje kak način `local` (na Linuxu brez GPU-ja CPU verzijo),
- zažene Qdrant v Dockerju ali, če Dockerja ni, kot samostojen program v `.qdrant/`
  (oboje samo na `127.0.0.1`),
- preveri, ali ima Ollama na `OLLAMA_URL` potrebna modela, in ju po potrebi prenese.

Primer `.env` (Ollama že teče na strežniku):

```bash
OLLAMA_URL=http://127.0.0.1:11434
QDRANT_URL=http://127.0.0.1:6340
RAG_API_HOST_PORT=8010
HOST=127.0.0.1          # dostopno samo s strežnika, uporabi SSH tunel
```

### 3. Dostop

Na deljenem strežniku porta ne odpiraj javno, uporabi SSH tunel:

```bash
ssh -L 8010:localhost:8010 streznik
```

in odpri **http://localhost:8010** lokalno.

## Ostale nastavitve (`.env`)

Vse nastavitve s komentarji so v `.env.example`. Poleg zgoraj opisanih:

| Spremenljivka | Privzeto | Namen |
|---|---|---|
| `TOP_K` | `4` | Koliko kosov besedila dobi LLM pri vsakem vprašanju. Več = več konteksta, a počasneje. V spletnem vmesniku ga lahko spremeniš za posamezno vprašanje (polje Top-k) |
| `QDRANT_URL` | prazno | Prazno pomeni kontejner `qdrant`. Nastavi za zunanji Qdrant |
| `QDRANT_COLLECTION` | `Beees` | Ime kolekcije. Nova kolekcija je alternativa `--reset` |
| `AUTO_INGEST_ENABLED` | `true` | Ob zagonu s prazno bazo samodejno naloži `AUTO_INGEST_DIR` |
| `AUTO_INGEST_DIR` | `dataset/data` | Mapa za samodejno nalaganje (`dataset/questions` vsebuje vprašanja za evalvacijo in ne sodi v bazo) |
| `LLM_MAX_NEW_TOKENS` | `512` | Največja dolžina odgovora |
| `LLM_TIMEOUT` | `300` | Časovna omejitev za LLM v sekundah |
| `LLM_THINK` | `false` | Ollama: modeli, ki "razmišljajo" (qwen3, qwen3.5), najprej razmislijo. Boljše pri težjih vprašanjih, a večkrat počasneje |
| `MODELS_GPU` | `auto` | GPU za kontejner `models`: `auto`, `true`, `false` |
| `BIND_HOST` | `0.0.0.0` | Na katerem naslovu poslušajo porti kontejnerjev. `127.0.0.1` = samo lokalno |
| `HOST` | `0.0.0.0` | Enako za `start-local.sh` |
| `RAG_API_HOST_PORT` / `QDRANT_HOST_PORT` / `OLLAMA_HOST_PORT` | `8000` / `6333` / `11434` | Spremeni, če je port zaseden |

## Težave

**`port is already allocated`**: port je zaseden. V `.env` nastavi drug `RAG_API_HOST_PORT`
(ali `QDRANT_HOST_PORT` / `OLLAMA_HOST_PORT`) in znova zaženi.

**`permission denied ... docker.sock`**: uporabnik ni v skupini `docker`.
Zaženi `sudo usermod -aG docker $USER` in se znova prijavi. Skripte do takrat same uporabijo `sudo`.

**Napaka o neujemanju dimenzij**: zamenjal si embedder. Zaženi `scripts/init.sh --reset`.

**Odgovori so samo odlomki (extractive)**: LLM ni dosegljiv. Preveri z `scripts/init.sh --check`
in `docker compose logs models`.

**`RERANK_MODE=local needs sentence-transformers`**: nastavi `LOCAL_MODELS=true` in zaženi
`scripts/start.sh`. Hitreje je `RERANK_MODE=api` (reranker na GPU-ju v kontejnerju `models`).

**`reranker failed, using unreranked order`** v logih `web`: reranker še nalaga model ali ne
teče. Preveri `docker compose logs models | grep reranker`.

**Modeli tečejo na CPU-ju, čeprav je GPU**: `docker info | grep -i nvidia` ne izpiše nič, torej
manjka NVIDIA Container Toolkit, ali pa se gonilnik ne ujema z jedrom (`nvidia-smi` javi napako).
Pri starejših karticah (npr. GTX 1080 Ti) uporabi gonilnik 580; novejši jih ne podpirajo.

**`Connection refused` do Qdranta**: klici se samodejno ponovijo, heartbeat pa beleži izpade:
`docker compose logs -f web | grep heartbeat`.

**"Internal Server Error" v brskalniku**: uporabi `scripts/chat.sh`, ki izpiše celotno napako.
