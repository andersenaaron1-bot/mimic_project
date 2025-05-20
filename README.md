# Installation
## Using Poetry
Poetry is a dependency and package manager which can easily resolve dependency conflicts (i.e., it can combine the currect python package versions).
### Step 1 - Install Poetry
Install poetry for your system: https://python-poetry.org/docs/

*Note: It is intended that poetry is installed in a folder that is separate from your global python distribution*

**Windows (Powershell)**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```
**Verify that poetry is available globally**
```powershell
poetry --version
```

### Step 2 - Create a Virtual Environment with Poetry
1. Open the terminal
2. Go to the path where the pyproject.toml is located (e.g., C.\Repositories\mimic_project)
3. Run 
```powershell
poetry install
```
4. Note the path to poetry
```powershell
poetry env info --path
```

### Step 3 - Set up in Pycharm
1. Open the project root in Pycharm (e.g., mimic_project)
2. Choose the path to python (python.exe) in the virtual poetry environment as "existing interpreter"


## Adding Packages to Project
Use poetry in Pycharm for this:
```powershell
poetry add [name_of_package]
```