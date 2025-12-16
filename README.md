# Snifferon

## Creating a new folder on `main`

1. Switch to the main branch locally:
   ```bash
   git checkout main
   ```
2. Create the folder **and add at least one file** so Git can track it (replace `my-feature` with your folder name):
   ```bash
   mkdir my-feature
   touch my-feature/README.md  # or add your actual files here
   ```
   Git tracks files, not empty directories, so every folder you commit needs at least one file inside.
3. Stage and commit the new folder (with its files):
   ```bash
   git add my-feature
   git commit -m "Add my-feature folder"
   git push origin main
   ```

Alternatively, from the GitHub web UI:
1. Go to the repository’s **main** branch.
2. Click **Add file**, then **Create new file**, and type `my-feature/README.md` (or another file name) to create the folder.
3. Add any starter content, then commit the change to `main`.
