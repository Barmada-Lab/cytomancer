import click

from cytomancer.config import config
from .helpers import test_cvat_credentials


@click.command("auth")
@click.option("--cvat-username", prompt="CVAT Username")
@click.password_option("--cvat-password", prompt="CVAT Password", confirmation_prompt=False)
def cvat_auth(cvat_username, cvat_password):
    """
    Update CVAT credentials. Run this with no arguments to get an interactive prompt that hides your password.
    """

    print(f"\nTesting CVAT connection to server {config.cvat_url}...")
    if not test_cvat_credentials(config.cvat_url, cvat_username, cvat_password):
        print("Connection failed. Please verify your credentials and try again.")
        print("See `cyto config update --help` for other CVAT-related settings")
        return

    print("Authentication successful. Saving credentials.")
    config.cvat_username = cvat_username
    config.cvat_password = cvat_password
    config.save()


def register(cli: click.Group):
    @cli.group("cvat", help="Tools for working with CVAT")
    @click.pass_context
    def cvat_group(ctx):
        ctx.ensure_object(dict)

    from cytomancer.cvat.survival import cli_entry as cvat_survival
    from cytomancer.cvat.upload2 import cli_entry
    from cytomancer.cvat.nuc_cyto import cli_entry as cvat_nuc_cyto

    cvat_group.add_command(cvat_auth)
    cvat_group.add_command(cvat_survival)
    cvat_group.add_command(cli_entry)
    cvat_group.add_command(cvat_nuc_cyto)
