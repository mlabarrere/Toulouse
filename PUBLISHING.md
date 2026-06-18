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

### 2. Ajouter le Trusted Publisher sur PyPI

Le Trusted Publishing (OIDC) ne nécessite **aucun token API** : l'authentification se fait
automatiquement entre GitHub Actions et PyPI.

1. Sur PyPI, allez dans **Account settings** → **Publishing** → **Add a new pending publisher**
   (ou, pour un projet existant : **Manage** → **Publishing**).
2. Remplissez les informations exactement :
   - **PyPI Project Name** : `toulouse`
   - **Owner** : `mlabarrere`
   - **Repository name** : `Toulouse`
   - **Workflow name** : `publish-PyPi.yml`
   - **Environment name** : `pypi`
3. Cliquez sur **Add**.

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

Le workflow `.github/workflows/publish-PyPi.yml` se déclenche automatiquement lorsque :

- un **tag** `v*` est poussé (ex. `git tag v1.1.2 && git push origin v1.1.2`), ou
- une **Release** GitHub est publiée.

Il construit la distribution puis la publie sur PyPI via Trusted Publishing (sans token).
Pensez d'abord à incrémenter la version :

```bash
# Met à jour la version dans pyproject.toml, commit et crée le tag
bump2version patch   # ou minor / major
git push --follow-tags
```

### Via ligne de commande

```bash
# Construire puis publier (nécessite des identifiants PyPI configurés)
uv build --no-sources
uv publish
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