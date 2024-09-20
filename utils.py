
# lendo o arquivo e passando para o array
def load_class_names(names_file):
    with open(names_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names
