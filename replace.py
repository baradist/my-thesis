import glob
import re


# root_dir needs a trailing slash (i.e. /root/dir/)
root_dir = './'


def replace_pattern(text):
    pattern = '\\\\textit\{([А-Яа-яёЁa-zA-Z\s,\(\)\-\d]+)\}'
    repl = r'\1'

    return re.sub(pattern, repl, text)


def replace(filename):
    with open(filename, encoding='utf-8') as src:
        with open(filename + '_', 'w', encoding='utf-8') as dest:
            text = src.read()
            replaced = replace_pattern(text)
            dest.write(replaced)
            dest.close()


for filename in glob.iglob(root_dir + '**/*.tex', recursive=True):
    print(filename)
    replace(filename)
