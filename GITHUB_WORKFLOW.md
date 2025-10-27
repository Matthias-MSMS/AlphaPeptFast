# GitHub Workflow for Claude Code

## Creating New GitHub Repositories

Claude Code can create GitHub repositories directly using the `gh` CLI:

```bash
gh repo create Matthias-MSMS/RepoName \
  --public \
  --source=. \
  --remote=origin \
  --push \
  --description "Repository description"
```

**Options:**
- `--public` or `--private` - Repository visibility
- `--source=.` - Use current directory as source
- `--remote=origin` - Set as git remote named "origin"
- `--push` - Push initial commit immediately
- `--description "..."` - Optional repository description

**Your GitHub username:** `Matthias-MSMS`

## Typical Workflow

1. **Initialize local repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit message"
   ```

2. **Create GitHub repository and push:**
   ```bash
   gh repo create Matthias-MSMS/RepoName --public --source=. --remote=origin --push
   ```

3. **View repository:**
   ```
   https://github.com/Matthias-MSMS/RepoName
   ```

This automatically handles authentication and sets up remote tracking.

## Common Git Operations

### Committing Changes
```bash
git add .
git commit -m "Commit message"
git push
```

### Creating Pull Requests
```bash
git checkout -b feature-branch
# Make changes
git add .
git commit -m "Add new feature"
git push -u origin feature-branch
gh pr create --title "Feature title" --body "Description"
```

### Checking Status
```bash
git status
git log --oneline -10
```
