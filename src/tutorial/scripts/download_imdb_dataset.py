import click
import logging
import tarfile
from tensorflow.keras import utils


@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    """ Downloads the IMDb Large Movie Review dataset to specified path.

    # References
        Mass et al., http://www.aclweb.org/anthology/P11-1015
    """
    logger = logging.getLogger(__name__)
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    logger.info('downloading dataset or retrieving from cache')
    path = utils.get_file(fname="aclImdb_v1.tar.gz",
                          origin=url)
    logger.info('extracting tar file')
    tar = tarfile.open(path, 'r:gz')
    tar.extractall(path=output_filepath)
    tar.close()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
