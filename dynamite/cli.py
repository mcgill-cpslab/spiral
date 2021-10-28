"""Console script for dynamite."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("dynamite")
    click.echo("=" * len("dynamite"))
    click.echo("A tool to help building ML pipeline easier for non technical users.")


if __name__ == "__main__":
    main()  # pragma: no cover
