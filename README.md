# structure-optimization

Optimization of physical structures

## Usage

Run

```bash
streamlit run app.py
```

## Development

### Pre-commit

Run

```bash
pre-commit run --all-files
```

to run all pre-commit hooks, including style formatting and unit tests.

### Package management

Update [`requirements.in`](requirements.in) with new direct dependencies.

Then run

```bash
pip-compile requirements.in
```

to update the [`requirements.txt`](requirements.txt) file with all indirect and transitive dependencies.

Then run

```bash
pip install -r requirements.txt
```

to update your virtual environment with the packages.

## Credits

Based on [*"A 165 Line Topology Optimization Code"* by Niels Aage & Villads Egede Johansen, January 2013](https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python).
