import io


def write_to_filepath(text, filepath):
    with io.open(filepath, 'w', encoding='iso-8859-15') as f:
        f.write(text)
