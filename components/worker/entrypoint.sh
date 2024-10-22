#!/bin/bash

source .venv/bin/activate

celery -A cytomancer worker -P solo -l info
