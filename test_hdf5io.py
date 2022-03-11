import pytest
from contextlib import closing
from .hdf5io import Hdf5io
import numpy
import tempfile
import sys


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

def test_dict2hdf():
    import copy
    from .introspect import dict_isequal
    data = {'a': 10, 'b': 5 * [3]}
    data = 4 * [data]
    data2 = {'a': numpy.array(10), 'b': numpy.array(5 * [3])}
    h = Hdf5io(filename, filelocking=False)
    h.data = copy.deepcopy(data)
    h.data2 = copy.deepcopy(data2)
    h.save(['data', 'data2'])
    del h.data
    h.load(['data', 'data2'])
    h.close()
    # hdf5io implicitly converts list to ndarray
    assertTrue(dict_isequal(data2, h.data2) and data == h.data)

def test_recarray2hdf():
    import copy
    data = numpy.array(list(zip(list(range(10)), list(range(10)))),
                       dtype={'names': ['a', 'b'], 'formats': [numpy.int, numpy.int]})
    data = 4 * [data]
    h = Hdf5io(filename, filelocking=False)
    h.data = copy.deepcopy(data)
    h.save(['data'])
    del h.data
    h.load(['data'])
    h.close()
    assertTrue((numpy.array(data) == numpy.array(h.data)).all())

def test_findvar():
    f = os.path.join(unit_test_runner.TEST_reference_data_folder,
                     'fragment_-373.7_-0.8_-160.0_MovingDot_1331897433_3.hdf5')
    h5f = Hdf5io(f, filelocking=False)
    h5f.findvar('MovingDot_1331897433_3')
    res = h5f.findvar(['position', 'machine_config'])
    pass

def test_complex_data_structure():
    item = {}
    item['a1'] = 'a1'
    item['a2'] = 2
    item['a3'] = 5
    items = 5 * [item]
    f = filename
    h5f = Hdf5io(f, filelocking=False)
    h5f.items = items
    h5f.save('items', verify=True)
    h5f.close()
    reread = read_item(f, 'items', filelocking=False)
    assertEqual(items, reread)


def test_listoflists():
    items = [['1.1', '1.2', '1.3'], ['2.1', '2.2']]
    f = filename
    h5f = Hdf5io(f, filelocking=False)
    h5f.items = items
    h5f.save('items')
    h5f.close()
    assertEqual(items, read_item(f, 'items'), filelocking=False)

