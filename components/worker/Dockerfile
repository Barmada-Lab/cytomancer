# install deps
FROM python:3.11.8-bookworm as builder
WORKDIR /app

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY pyproject.toml poetry.lock .

RUN mkdir cytomancer; touch cytomancer/__init__.py

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install

RUN rm -rf cytomancer

# copy cached deps and latest source
FROM python:3.11.8-bookworm as stripped

COPY --from=builder /app/.venv /app/.venv

# finally, get the latest source
FROM stripped as runtime
WORKDIR /app

COPY ./cytomancer ./cytomancer

COPY components/worker/entrypoint.sh .

ENTRYPOINT [ "./entrypoint.sh" ]
