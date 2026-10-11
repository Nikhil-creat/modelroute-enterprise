# Publish the ModelRoute Enterprise site on GitHub Pages

The website lives in `docs/` (static HTML, no build step). Your trained models and data never go to GitHub:
`data/` and `artifacts/` are in `.gitignore`.

## A. From your phone with Termux
```bash
pkg install git unzip
termux-setup-storage
cd ~/storage/downloads
unzip modelroute_enterprise_master.zip
cd modelroute_enterprise
git init
git add .
git commit -m "ModelRoute Enterprise"
git branch -M main
git remote add origin https://github.com/Nikhil-creat/modelroute-enterprise.git
git push -u origin main
```
1. Create the empty repo first: github.com → **New repository** → name `modelroute-enterprise` → Public → Create (no README).
2. When `git push` asks for a password, paste a **Personal Access Token** (GitHub → Settings → Developer settings → Tokens (classic) → scope `repo`).

## B. Without Termux (browser only)
1. Create the repo, tap **Add file → Upload files**.
2. Upload the contents of `docs/` (keep the `docs/data/` folder and the `.nojekyll` file) and commit.

## Turn on Pages
1. Repo → **Settings → Pages**.
2. **Source:** Deploy from a branch → **Branch:** `main` → **Folder:** `/docs` → **Save**.
3. Wait 1–2 minutes. Your site: `https://nikhil-creat.github.io/modelroute-enterprise/`

## Use the assistant and the live agent
1. Get a free key: console.groq.com → **API Keys** → Create.
2. Open your site → tap **💬 Assistant** → ⚙ → paste the key → Save.
3. The key is stored only in your browser (localStorage) and sent only to api.groq.com. Never commit a key.

## Update the benchmark table
After running `python -m evaluation.benchmark`, commit `docs/data/benchmark.json` and push; the table fills in automatically.

## Troubleshooting
- 404 page: check Settings → Pages shows `main` and `/docs`, and that `docs/index.html` exists on GitHub.
- Assistant says "Invalid API key": re-paste the key (no spaces) in ⚙.
- Old content after a push: hard refresh, Pages can take a couple of minutes.
