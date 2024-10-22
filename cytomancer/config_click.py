import click

from cytomancer.config import config


@click.command("show")
def show_config():
    """
    Display current configuration settings.
    """
    print("\nCurrent settings:")
    for k, v in config.model_dump().items():
        if k == "cvat_password":
            v = "*" * len(v)
        print(f"\t{k}: {v}")
    print()


# Hacky little thing that adds options for all settings to set_config
def settings_options():
    def combined_decorator(func):
        for k, v in reversed(config.model_dump().items()):
            if k == "cvat_password" or k == "cvat_username":
                continue
            decorator = click.option(f"--{k}", default=v, show_default=True)
            func = decorator(func)
        return func

    return combined_decorator


@click.command("update")
@settings_options()
def update_config(**kwargs):
    """
    Update config
    """
    for k, v in kwargs.items():
        setattr(config, k, v)
    config.save()


def register(cli: click.Group):
    @cli.group("config", help="Config management")
    @click.pass_context
    def config_group(ctx):
        ctx.ensure_object(dict)

    config_group.add_command(show_config)
    config_group.add_command(update_config)
