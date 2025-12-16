# Snifferon

## Creating a new folder on `main`

1. Switch to the main branch locally:
   ```bash
   git checkout main
   ```
2. Create the folder **and add at least one file** so Git can track it:
   ```bash
   mkdir <folder-name>
   touch <folder-name>/README.md  # or add your actual files here
   ```
3. Stage and commit the new folder (with its files):
   ```bash
   git add <folder-name>
   git commit -m "Add <folder-name> folder"
   git push origin main
   ```

Alternatively, from the GitHub web UI:
1. Go to the repository’s **main** branch.
2. Click **Add file** ➜ **Create new file**, type `<folder-name>/README.md` (or another file name) to create the folder.
3. Add any starter content, then commit the change to `main`.
