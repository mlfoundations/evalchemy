# PostgreSQL Database

Currently the database is a PostgreSQL DB. We have separate tables for `datasets`, `models`, `evalresults`, and `evalsettings`. The specific schemas can be found in [models.py](models.py)

## Credentials
Most of the DB configs are outlined in [config.py](config.py). The only additional credential you need is the password, which can be set using `export DB_PASSWORD=(password-here)`. 