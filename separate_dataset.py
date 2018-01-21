import data_handler

dataset = data_handler.DataSet(
    'data/REFERENCE.csv', data_handler.load_composed,
    path='data/',
    #remove_noise=True, tokens='NAO'
    )
train_set, test_set = dataset.disjunct_split(.9)

train_set.save('data/train.csv')
test_set.save('data/test.csv')
