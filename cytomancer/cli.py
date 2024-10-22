import logging

import click
from trogon import tui

from cytomancer.config_click import register as register_config
from cytomancer.cvat.click import register as register_cvat
from cytomancer.fiftyone.click import register as register_fiftyone
from cytomancer.oneoffs.click import register as register_oneoffs
from cytomancer.quant.click import register as register_quant

logger = logging.getLogger(__name__)


@tui()
@click.group(help="Cytomancer CLI")
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)


register_quant(cli)
register_cvat(cli)
register_oneoffs(cli)
register_config(cli)
register_fiftyone(cli)
