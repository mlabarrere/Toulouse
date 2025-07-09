# Guide de publication avec uv

Ce guide explique comment publier le package `toulouse` sur PyPI en utilisant `uv` et le trusted publishing.

## Prérequis

1. **Compte PyPI** : Créez un compte sur [PyPI](https://pypi.org/account/register/)
2. **Repository GitHub** : Votre code doit être sur GitHub
3. **Environnement GitHub** : Configurez un environnement "pypi" dans les paramètres GitHub

## Configuration du Trusted Publishing

### 1. Créer l'environnement GitHub

1. Allez dans votre repository GitHub
2. Cliquez sur **Settings** → **Environments**
3. Cliquez sur **New environment**
4. Nommez-le `pypi`
5. Cliquez sur **Configure environment**

### 2. Configurer PyPI

1. Allez sur [PyPI](https://pypi.org/manage/account/)
2. Cliquez sur **API tokens** dans le menu de gauche
3. Cliquez sur **Add API token**
4. Sélectionnez **Entire account (all projects)**
5. Copiez le token généré

### 3. Ajouter le Trusted Publisher

1. Sur PyPI, allez dans **Account settings** → **Trusted publishers**
2. Cliquez sur **Add**
3. Remplissez les informations :
   - **Publisher name** : `toulouse-publisher`
   - **Owner** : `votre-username-github`
   - **Repository name** : `toulouse`
   - **Workflow name** : `Upload Python Package`
   - **Environment name** : `pypi`
4. Cliquez sur **Add trusted publisher**

## Test local

Avant de publier, testez localement :

```bash
# Construire le package
uv build --no-sources

# Vérifier le contenu
ls dist/

# Tester l'installation
uv run --with toulouse --no-project -- python -c "import toulouse; print('Success!')"
```

## Publication

### Via GitHub Actions (recommandé)

1. Créez un **Release** sur GitHub
2. Le workflow se déclenche automatiquement
3. Le package est publié sur PyPI

### Via ligne de commande

```bash
# Publier directement
uv publish

# Ou avec un token spécifique
uv publish --token YOUR_TOKEN
```

## Vérification

Après publication, vérifiez que le package est disponible :

```bash
# Installer depuis PyPI
pip install toulouse

# Tester l'import
python -c "import toulouse; print(toulouse.__version__)"
```

## Dépannage

### Erreur "Environment not found"
- Vérifiez que l'environnement "pypi" existe dans les paramètres GitHub
- Assurez-vous que le workflow a les bonnes permissions

### Erreur "Trusted publisher not found"
- Vérifiez que le trusted publisher est configuré correctement sur PyPI
- Assurez-vous que le nom du workflow correspond exactement

### Erreur de build
- Vérifiez que `pyproject.toml` est correctement configuré
- Testez localement avec `uv build --no-sources`

## Avantages de uv publish

- ✅ **Plus simple** : Une seule commande `uv publish`
- ✅ **Plus sécurisé** : Trusted publishing sans tokens
- ✅ **Plus rapide** : Build et publish en une étape
- ✅ **Plus moderne** : Utilise les standards actuels 