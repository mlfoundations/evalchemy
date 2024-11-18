# PostgreSQL Database

Currently the database is a PostgreSQL DB. We have separate tables for `datasets`, `models`, `evalresults`, and `evalsettings`. The specific schemas can be found in [models.py](models.py)

## Credentials
Most of the DB configs are outlined in [config.py](config.py). You will need to supply your DB_HOST and DB_PASSWORD, which can be set using `export DB_HOST=(db-address-here)` and `export DB_PASSWORD=(db-password-here)`. 