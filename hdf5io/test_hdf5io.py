import pytest
from .hdf5io import Hdf5io, read_item
import numpy

@pytest.fixture()
def temp_hdf_handler():
    """
    Generate temporary empty hdf file and return its handler. Delete file after test is done.
    Returns
    -------

    """
    import tempfile
    import os
    import time
    path = tempfile.mkdtemp()
    fp = os.path.join(path,f'{time.time()}.hdf5')
    with Hdf5io(fp,lockfile_path=tempfile.gettempdir()) as handler:
        yield handler
    os.remove(fp)
    if not os.listdir(path):
        os.rmdir(path)

def test_locking(temp_hdf_handler):
    """
    Tests if a locked file can be opened while being open via the test fixture hdf handler.
    """
    temp_hdf_handler.save('filename')
    with Hdf5io(temp_hdf_handler.filename, lockfile_path=temp_hdf_handler.lockfile_path) as handler:
        handler.dummy='test'
        handler.save('dummy')
    temp_hdf_handler.close()
    #with pytest.raises(PermissionError) as e_info:
    read_item(temp_hdf_handler.filename, 'filename', lockfile_path=temp_hdf_handler.lockfile_path)
    assert True
    # all these pass as there is no real concurrency happening
    # TODO: multithreaded read/write in a loop that provokes a race condition

def test_saveload_pickled(temp_hdf_handler):
    """
    Tests saving/loading pickled byte array.
    """
    import pickle
    myarray = numpy.random.randn(2,4)
    temp_hdf_handler.pstream = pickle.dumps(myarray)
    temp_hdf_handler.save('pstream')
    del temp_hdf_handler.pstream
    temp_hdf_handler.load('pstream')
    myarray2 = pickle.loads(temp_hdf_handler.pstream)
    numpy.testing.assert_almost_equal(myarray, myarray2, 1)

def test_masked_array(temp_hdf_handler):
    ma0 = numpy.ma.array([1, 2, 3], dtype=float)
    ma0.mask = [1, 0, 1]
    numpy.ma.set_fill_value(ma0, numpy.nan)
    madict = {'anarray': ma0.copy()}
    temp_hdf_handler.ma = ma0
    temp_hdf_handler.madict = madict
    temp_hdf_handler.save(['ma', 'madict'])
    del temp_hdf_handler.ma
    del temp_hdf_handler.madict
    ma = temp_hdf_handler.findvar('ma')
    madict1 = temp_hdf_handler.findvar('madict')
    numpy.testing.assert_array_almost_equal(temp_hdf_handler.ma, ma0)
    for k1 in madict1:
        numpy.testing.assert_array_almost_equal(madict[k1], madict1[k1])

def test_dict2hdf(temp_hdf_handler):
    import copy
    from .introspect import dict_isequal
    data = {'a': 10, 'b': 5 * [3]}
    data = 4 * [data]
    data2 = {'a': numpy.array(10), 'b': numpy.array(5 * [3])}
    h = temp_hdf_handler
    h.data = copy.deepcopy(data)
    h.data2 = copy.deepcopy(data2)
    h.save(['data', 'data2'])
    del h.data
    h.load(['data', 'data2'])
    h.close()
    # hdf5io implicitly converts list to ndarray
    assert (dict_isequal(data2, h.data2) and data == h.data)

def test_recarray2hdf(temp_hdf_handler):
    import copy
    data = numpy.array(list(zip(list(range(10)), list(range(10)))),
                       dtype={'names': ['a', 'b'], 'formats': [numpy.int, numpy.int]})
    data = 4 * [data]
    temp_hdf_handler.data = copy.deepcopy(data)
    temp_hdf_handler.save(['data'])
    del temp_hdf_handler.data
    temp_hdf_handler.load(['data'])
    temp_hdf_handler.close()
    numpy.testing.assert_array_equal(numpy.array(data), numpy.array(temp_hdf_handler.data))

def test_complex_data_structure(temp_hdf_handler):
    item = {}
    item['a1'] = 'a1'
    item['a2'] = 2
    item['a3'] = 5
    items = 5 * [item]
    temp_hdf_handler.items = items
    temp_hdf_handler.save('items', verify=True)
    temp_hdf_handler.close()
    reread = read_item(temp_hdf_handler.filename, 'items')
    assert items==reread


def test_listoflists(temp_hdf_handler):
    items = [['1.1', '1.2', '1.3'], ['2.1', '2.2']]
    h5f = temp_hdf_handler
    h5f.items = items
    h5f.save('items')
    h5f.close()
    assert items == read_item(h5f.filename, 'items', lockfile_path=h5f.lockfile_path)

