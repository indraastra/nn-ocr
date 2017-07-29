import click

from dataset import cjk
from dataset import en
from dataset.dataset import load, save, preview


DATASETS = {
    en.DATASET_NAME: en,
    cjk.DATASET_NAME: cjk
}

@click.group()
def cli():
    pass


@cli.command()
@click.argument('output_dir')
@click.option('--dataset', default=en.DATASET_NAME)
@click.option('--image_size', default=72)
@click.option('--font_limit', default=-1)
def save_data(output_dir, dataset, image_size, font_limit):
    dataset = DATASETS[dataset]
    labels = dataset.get_labels()
    (X_train, y_train), (X_test, y_test) = load(dataset.get_labels(),
            dataset.get_fonts(), image_size, font_limit=font_limit)
    dataset_dir = os.path.join(output_dir, dataset)
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

    save(X_train, y_train, labels, train_dir)
    save(X_test, y_test, labels, test_dir)


@cli.command()
@click.option('--dataset', default=en.DATASET_NAME)
@click.option('--image_size', default=72)
@click.option('--cells_x', default=10)
@click.option('--cells_y', default=10)
@click.option('--font_limit', default=-1)
@click.option('--label', default=None)
@click.option('--randomize', default=False)
def preview_data(dataset, image_size, cells_x, cells_y, font_limit, label, randomize):
    dataset = DATASETS[dataset]
    labels = dataset.get_labels()
    (X_train, y_train), (X_test, y_test) = load(dataset.get_labels(),
            dataset.get_fonts(), image_size, font_limit=font_limit)
    if label:
        click.echo('Previewing en data for label: {}'.format(label))
        label_idx = labels.index(label)
        X_preview = X_train[y_train == label_idx]
        y_preview = y_train[y_train == label_idx]
    else:
        click.echo('Previewing en data for all labels')
        X_preview = X_train
        y_preview = y_train
    preview(X_preview, y_preview, labels, (cells_x, cells_y), randomize)


if __name__ == '__main__':
    cli()
