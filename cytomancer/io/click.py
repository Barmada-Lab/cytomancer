import click

from .dump import dump_tiffs_cli


def register(cli: click.Group):
    @cli.group("io", help="I/O utilities")
    @click.pass_context
    def io_group(ctx):
        ctx.ensure_object(dict)

    io_group.add_command(dump_tiffs_cli)
