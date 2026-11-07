# The executor for P4: slim Python plus the libraries a data-analysis agent actually writes.
#
# `python:3.12-slim` is the wrong default and the run of 2026-11-07 proved it: all six programs
# opened with `import pandas`, and all six died with ModuleNotFoundError - one hundred percent of
# the episodes were scored against a library that was not there.  A sandbox that cannot run the
# obvious program is not a neutral environment; it is a reward source that says "wrong" to every
# attempt, and only the record's sandbox_errors column makes that visible.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
        pandas==2.2.3 \
        numpy==2.1.3

# The table is mounted at /w; the program is /w/prog.py.
WORKDIR /w
