# Creating the GitHub Repository

## Option 1: Using GitHub CLI (gh)

If you have GitHub CLI installed:

```bash
# Login to GitHub (if not already logged in)
gh auth login

# Create the repository
gh repo create model-garden --public --source=. --remote=origin --push

# Or for a private repository
gh repo create model-garden --private --source=. --remote=origin --push
```

## Option 2: Using GitHub Web Interface

1. **Go to GitHub**: https://github.com/new

2. **Repository Settings**:
   - Repository name: `model-garden`
   - Description: "Fine-tune and serve LLMs with carbon footprint tracking"
   - Visibility: Public (or Private)
   - ❌ Don't initialize with README, .gitignore, or license (we already have these)

3. **Click "Create repository"**

4. **Push your local repository**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/model-garden.git
   git branch -M main
   git push -u origin main
   ```

## Option 3: Using Git Commands Only

```bash
# Create repository on GitHub first (through web interface)
# Then connect your local repository:

git remote add origin https://github.com/YOUR_USERNAME/model-garden.git
git branch -M main
git push -u origin main
```

## Next Steps After Creating Repository

### 1. Configure Repository Settings

Go to repository Settings on GitHub:

- **About**: Add description and topics
  - Description: "Fine-tune and serve LLMs with carbon footprint tracking"
  - Topics: `llm`, `machine-learning`, `fine-tuning`, `carbon-tracking`, `sustainability`, `ai`, `python`, `fastapi`, `unsloth`, `vllm`

- **Features**: Enable
  - ✅ Issues
  - ✅ Discussions
  - ✅ Sponsorships (optional)

### 2. Add Branch Protection (Optional)

Settings → Branches → Add branch protection rule:
- Branch name pattern: `main`
- ✅ Require pull request reviews before merging
- ✅ Require status checks to pass before merging
- ✅ Require conversation resolution before merging

### 3. Set Up GitHub Actions (Future)

Create `.github/workflows/ci.yml` for automated testing:

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install uv
        uv pip install -e ".[dev,test]"
    
    - name: Run tests
      run: pytest --cov=model_garden
    
    - name: Lint
      run: ruff check .
    
    - name: Type check
      run: mypy model_garden
```

### 4. Add Issue Templates (Optional)

Create `.github/ISSUE_TEMPLATE/`:
- `bug_report.md`
- `feature_request.md`

### 5. Update README

After creating the repository, update the README with your actual GitHub username:

```bash
# Replace placeholders in README
sed -i '' 's/yourusername/YOUR_ACTUAL_USERNAME/g' README.md
git add README.md
git commit -m "docs: update GitHub username in README"
git push
```

## Verify Your Repository

After pushing, verify on GitHub:
- ✅ All files are present
- ✅ README displays correctly
- ✅ Documentation is readable
- ✅ License is recognized

## Share Your Project

Once live, you can share:
- GitHub URL: `https://github.com/YOUR_USERNAME/model-garden`
- Clone URL: `git clone https://github.com/YOUR_USERNAME/model-garden.git`
- Social media with relevant hashtags: #LLM #MachineLearning #AI #Sustainability
