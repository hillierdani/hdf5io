'''Collection of methods that work on (list of) strings'''
from future.utils import string_types
import re
import numpy
import os

timestamp_re = re.compile('.*((?<!\d)\d{15,15}|(?<!\d)\d{10,10}).*')

def is_decodable_to_str(bytearray_in, codec = 'Latin1'):
    ''' Strings stored by pytables are read back as byte array. This one liner detects if the byte array can be converted back to string or not'''
    if is_pickle_bytestream(bytearray_in):
        return False # pickle stream should not be decoded
    return hasattr(bytearray_in, 'decode') and isinstance(bytearray_in.decode(codec), str)

def coerce_if_string23(str_or_byte):
    ''' Handles all cases when str_or_byte is a string or can be converted to a string.
    Leaves data untouched when it is not a string. '''
    if is_pickle_bytestream(str_or_byte): return  # pickle stream should not be decoded
    if isinstance(str_or_byte, string_types): return str_or_byte
    if is_decodable_to_str(str_or_byte):
        try:
            return str_or_byte.decode()
        except:
            t=1
    if hasattr(str_or_byte, 'astype') and str_or_byte.dtype.kind == 'S': return str_or_byte.astype('U13')
    return str_or_byte


def is_pickle_bytestream(bytearray_in):
    ''' Guesses whether data (byte array) is a string or pickled stream'''
    is_pickle_stream = (isinstance(bytearray_in, (bytes, bytearray)) or hasattr(bytearray, 'dtype') and bytearray.dtype.kind == 'S')\
                       and bytearray_in[0:1] == b'\x80' and bytearray_in[1:2][-1] <=4 and bytearray_in[-1:]==b'.'
    return is_pickle_stream


def latex_float(f, only_exponent=False):
    '''Converts a python float into latex string, e.g. 0.001 -> 10^-3'''
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        if only_exponent:
            return r"10^{{{0}}}".format(int(exponent))
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def extract_common_string(flist, to_remove=['\)-r\d-']):
    ''' Locates '-rx-' part in filenames (where -rx- mean repetition x) and tries to find filenames that differ
    only in x but otherwise are identical,i.e. these files represent repeated trials of the stimulus. Returns the 
    strings without the varying parts
    '''
    flist2 = flist[:]
    for k in to_remove: # remove varying parts from the strings
        ss = re.compile(k)
        flist2= [ss.sub('', i) for i in flist2]
    flistu,  indices = numpy.unique(flist2, return_inverse=True)
    commons   =[]
    for i in numpy.unique(indices):
        commons.append([flist2[i1] for i1 in numpy.where(indices==i)[0]][0])
    return  commons

def join(*args):
    '''Same functionality as os.path.join but for paths using dot as separator'''
    items = [text.replace('.', '/') for text in args]
    joined = os.path.join(*items)
    return joined.replace('/', '.')
    
def split(dotted_path):
    slashed = dotted_path.replace('.', '/')
    path, name = os.path.split(slashed)
    return path.replace('/', '.'),  name
    

def array2string(inarray, formatstring = ''):
    if hasattr(inarray,'ndim') and inarray.ndim == 2:
        a = ["%.3g "*inarray.shape[1] % tuple(x) for x in inarray]
    elif isinstance(inarray, (list,tuple)) or (hasattr(inarray,'ndim') and inarray.ndim == 1):
        fstr = '{0:'+formatstring +'}'
        a = [fstr.format(x) for x in inarray]
    return numpy.array(a)

