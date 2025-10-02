'''
The hdf5io module is a class that should be inherited. It provides methods to automatically save, load data from a hdf5 file 
and recreate derived data types.
'''
import sys, pdb
import pickle
try:
    import queue
except:
    import Queue as queue
import zc.lockfile
import numpy
import re
import hdf5io.introspect as introspect
from .stringop import join, coerce_if_string23, is_decodable_to_str
import os
import numbers
import tables
import time
import traceback
import logging
import threading
from contextlib import contextmanager, closing
log = logging.getLogger('hdf5io')
import warnings
warnings.filterwarnings('ignore',category=tables.NaturalNameWarning)
try: 
    import psutil    
    RAMLIMIT = psutil.virtual_memory().available*0.7
except:
    RAMLIMIT = 2*10**9 #2Gb
    pass
import tempfile
lockfilepath = tempfile.gettempdir()

class GlobalThreadLocks(threading.Thread):
    '''Class that keeps a global dict of lock objects. Locks have to be created with the create method since dict is not thread safe
    so we mush ensure there are no concurrent write operations to the lock dict. Once there is an entry with a lock it can safely be read
    by other threads to share the lock object. So there is a small speed penalty when creating the lock entry but virtually no speed sacrifice
    when accessing the lock.'''
    
    def __init__(self):
        self.lockpool = {}
        self.queue = queue.Queue()
        self.terminatesignal = object()
        threading.Thread.__init__(self)
        self.daemon=True
    
 #  def __del__(self): just let thread be terminated when main process stops?
  #      self.queue.put(self.terminatesignal)
        
    def run(self):
        while 1:
            command = self.queue.get(True)
            if command is self.terminatesignal: break
            if command[0] =='create': # (create, filename, lock object, threading.event)
                self.lockpool[command[1]] = command[2]
           # elif command[0] =='remove': #ever use this?
              #  del self.lockpool[command[1]]
                command[3].set()
    
    def create(self, key,  lockobject):
        if key in self.lockpool: 
           # print key+' lock alread exists'
            return
        signal = threading.Event()
        self.queue.put(('create', key, lockobject, signal))
        signal.wait()

class GlobalLock(object):
    def __init__(self):
        self.lock=threading.RLock()
        
    @contextmanager
    def acquire(self, block, timeout=60000): # 1 min timeout
        if block == False or timeout == 0:  # either lock should not block at all, or block with infinite waiting time
            if self.lock.acquire(block) == False:
                print('already locked, check who with traceback package')
                pdb.set_trace()
                raise RuntimeError('Cannot lock object,  it is already locked in another thread')
        else:  # timeout lock behavior
            cond = threading.Condition(threading.Lock())
            with cond:
                current_time = start_time = time.time()
                while current_time < start_time + timeout:
                    if self.lock.acquire(False):
                        break
                    else:
                        print(('could not acquire lock immediately, waiting {0}s'.format(timeout/1000)))
                        cond.wait(timeout - current_time + start_time)
                        current_time = time.time()
        yield
        self.lock.release()
        
    def release(self):
        self.lock.release()

if not hasattr(sys.modules[__name__], 'lockman'): #module might be imported multiple times but we only need a single class
    lockman = GlobalThreadLocks()
    lockman.start()

filters = tables.Filters(complevel=1, complib='blosc:lz4', shuffle=True)  # run benchmark in legacy.compressor_benchmark to find best compressor. This setting is fast decoding and similar file size as LZO. Fast decoding is critical since aggregation/comparer reads a lot into hdf5 files.
class Hdf5io(object):
    ''' 
    Handles data between RAM and hdf5 file. 
The create_ functions should not expect any parameters but have to access data from self so that these methods can be called in a loop.
    '''
    maxelem = 5 #number of digits defining how many items a list of dicts or list of lists can have. 5 means 999999
    elemformatstr = '{0:0'+str(maxelem)+'g}'

    def file_aware_getattr(self, item):
        ''' After opening hdf5 file overrides __getattr__ to automatically load missing attributes that are stored in the hdf5 file
         Checks if a variable is stored in hdf5 file. '''
        # This is only invoked if item is not yet an attribute of myinstance
        if item in self.h5f.list_nodes('/'):
            self.load(item)
            return getattr(self, item)
        else:
            raise AttributeError('Hdf5io cannot load or recreate {0} in file {1}'.format(item, self.filename))

    def __init__(self, filename, lockfile_path=lockfilepath, file_always_open=True, file_mode = 'a'):
        '''
        Opens/creates the hdf5 file for reading/writing.
        '''
        self.lockfile_path = lockfile_path
        if not hasattr(self, 'attrnames'):
            self.attrnames = []
        self.ramlimit = RAMLIMIT
        self.file_always_open = file_always_open
        filename = str(filename)  # not handling pathlib.Path objects (yet)
        if hasattr(filename,'shape') and filename.shape==(1,):
            filename = filename[0]
        if not filename[-4:] == 'hdf5':
            warnings.warn("HDF5 file name is expected to have .hdf5 extension")
        self.filename = filename
        cp, cf = os.path.split(self.filename)
        if not os.path.exists(cp):
            os.makedirs(cp)
        if 0: #os.path.exists(self.filename):
            try:
                import h5py
                openerror = not h5py.is_hdf5(self.filename)
                #tables.is_pytables_file(self.filename)
            except tables.HDF5ExtError as e1:
                # checks is file is usable, if not then renames current hdf5 file and starts a new one
                print(e1)
                openerror = 1
            if os.path.getsize(self.filename) < 50 or openerror:
                log.error(self.filename + "corrupted?")
                os.rename(self.filename, self.filename+'corrupted')
                raise RuntimeError(self.filename+' corrupted?')
        self.filelockkey = self.filename.replace('/', '\\')
        lockman.create(self.filelockkey, GlobalLock())
        self.lockdebug = False
        self.blocking_lock = True
        self.open_with_lock(file_mode)
     #   self.bind_finalizer('h5f')
        self.h5fpath = 'h5f.root'
        # rawdata is [row,col,chan,time] for legacy data acquired on RC setup, [row,col,time,chan] if ROIs have been detected via CaImAn.
        self.timeaxis=2
        if hasattr(self.h5f.root._v_attrs,'soma_rois_version') and 'caiman' in self.h5f.root._v_attrs.soma_rois_version:
            pass #raise NotImplementedError('Which axis is time after caiman conversion?') #else 3
        self.rawdatasource = join(self.h5fpath,'rawdata') #in memory, alternative is 'h5f.root' for disk mapped access
        self._want_abort = 0
        log.debug('Opening '+self.filename)
#        super(Hdf5io,self).__getattr__ = self.file_aware_getattr
#        self.command_queue=Queue.Queue()
   #     self.queueshutdown = object() #each thread has this object that has to be put in the command queue to stop the thread

    def __enter__(self):
        """
        Having this method defined enables using Hdf5io a context manager.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False  # does not suppress eventual exceptions

    def open_with_lock(self, filemode='a'):
        with lockman.lockpool[self.filelockkey].acquire(False):
            try:
                if self.lockfile_path:
                    if not os.path.exists(os.path.join(self.lockfile_path, 'filelocks')):
                        os.makedirs(os.path.join(self.lockfile_path, 'filelocks'))
                    hdf5filelock_filename = os.path.join(os.path.join(self.lockfile_path, 'filelocks', os.path.split(self.filename)[1][:-3]+'.lock'))
                    self.hdf5file_lock = zc.lockfile.LockFile(hdf5filelock_filename) # cross platform file lock, but puts an extra lock file in the file system
                try:
                    self.h5f = tables.open_file(self.filename, mode = filemode)
                except:
                    raise
                #list compression used to create CArrays in the file and check if pytables has the required library installed
                #the line below causes segfaul if hdf file is already open?
                #carray_complib = [item.filters.complib for item in self.h5f.iter_nodes(self.h5f.root,'CArray')] # walknodes is slow for file containing lots of nodes
                #if len(carray_complib)>0:
                 #   versions = [tables.which_lib_version(item) for item in carray_complib if item is not None]
                  #  if 0 and None in versions: # not in use but kept as a way to check complib versions
                   #     self.h5f.close()
                    #    if self.lockfile_path:self.hdf5file_lock.close()
                     #   raise RuntimeError('Compression library used for creating a CArray in this file is not available in the pytables installation on this computer')
            except zc.lockfile.LockError:
                with open(hdf5filelock_filename, 'r') as lockf:
                    lockf.seek(1)
                    pidinlock = lockf.read().strip()
                raise RuntimeError(str(os.getpid())+" cannot lock file "+self.filename+" that is currently locked by pid "+ pidinlock)
            except:
                traceback.print_exc()
                pdb.set_trace()
                raise RuntimeError(f"File {self.filename} cannot be opened, giving up.")
            finally:
                if hasattr(self, 'h5f') and self.h5f.isopen and not self.file_always_open:
                    self.h5f.close()
                if hasattr(self, 'hdf5file_lock') and self.lockfile_path: self.hdf5file_lock.close()
            
    
    def copy_file(self, newname, overwrite=False):
        '''copy, when file exists it will not overwrite'''
        if not overwrite and os.path.exists(newname):
            raise IOError('Destination hdf file exists and overwrite is set to False')
        newh = Hdf5io(newname, lockfile_path=self.lockfile_path) #opens with locking
        newh.h5f.close() #closes hdf5 file but does not release the lock
        self.h5f.copy_file(newname, overwrite=True) # this operation does not check the lock but 
        #we have it so, no problem. Set overwrite to true since we just created the destination hdf file to make sur#release the lock for the new file

    @property
    def timestamp(self):
        from .stringop import timestamp_re
        return int(timestamp_re.findall(self.filename)[0])

    @property
    @contextmanager
    def write(self):
        if self.lockdebug:
            print((os.path.split(self.filename)[1]+'write blocks:'+str(self.blocking_lock)))
        with lockman.lockpool[self.filelockkey].acquire(self.blocking_lock):#self.filelock:#.writelock:
            try:
                if not self.h5f.isopen: 
                    closeit=True
                    self.open_with_lock('a')
                    print(('write opened '+self.filename))
                else:
                    closeit = False
                if self.h5f.isopen and self.h5f.mode!='a':
                    self.h5f.close()
                    if self.lockfile_path: self.hdf5file_lock.close()
                    self.open_with_lock('a')
                    print(('write opened '+self.filename))
                    reopen = True
                else:
                    reopen = False
                yield
            except:
                raise
            finally:
                if closeit or reopen: 
                    print(('hdf5io write context closes file'+self.filename))
                    self.h5f.close()
                    if self.lockfile_path: self.hdf5file_lock.close()
                if reopen: 
                    self.open_with_lock('a')
                    print(('write reopened '+self.filename))
        

    @property        
    @contextmanager
    def read(self):
        #with self.filelock.readlock:
        if self.lockdebug:
            print((os.path.split(self.filename)[1]+'read blocks:'+str(self.blocking_lock)))
        with lockman.lockpool[self.filelockkey].acquire(self.blocking_lock):
            try:
                if not self.h5f.isopen: 
                    closeit=True
                    self.open_with_lock('a')
                    print(('read opened '+self.filename))
                else: 
                    closeit = False
                yield
            except:
                traceback.print_exc()
                raise
            finally: 
                if closeit: 
                    print('hdf5io read closes file')
                    self.h5f.close()
                    if self.lockfile_path: self.hdf5file_lock.close()
        
#    def __del__(self): #do not use: it is allowed to open the same file in multiple threads, locking ensures this is safe
   #     self.close()
    
    def __finalize__(self):
        try:
            self.h5f.close()
            print('hdf5io finalizer closes file')
        except Exception as detail:
            print(detail)
            print('finalizer could not close hdf5 file')
            
    def close(self):
        '''You must call this method if you want to close the hdf5 file'''
        self._want_abort = 1
        self.stacktrace = traceback.extract_stack()
  #      if self.is_alive():
     #       self.command_queue.put(self.queueshutdown)
       # else:
        log.debug("will abort "+self.filename)
        self.cleanup() # let running processes know that they must stop
    #print self.filename+' closed in hdf5io'
        
    
    def isUsable(self):
        '''returns true if hdf5 file is usable, i.e. not corrupted'''
        if not os.path.exists(self.filename):# or tables.is_pytables_file(self.filename) is None:
            a = self.h5f
            log.warning("HDF file not usable,  not exists or nor pytables file")
            return False
        elif os.path.exists(self.filename) and self.h5f.isopen and not hasattr(self.h5f.root, "rawdata"):
            a = self.h5f
            return False
        a = self.h5f
        return True

    def findvar(self, vnames,  stage=True, errormsg='',  overwrite=False, path = None, **kwargs):
        '''First checks if requested data is already in memory, if not it tries to load it from the hdf5 file,
        if data is neither in the hdf5 file, it tries to recreate from rawdata.
        This is the generic way of accessing data (excluding rawdata).
        Set stage to False if you do not want data be loaded from disk, just get the reference to it.'''
        debug = kwargs.get('debug', 0)
        if not isinstance(vnames,(tuple,list)):
            vnames = [vnames]
        mynodes = []
        for vname in vnames:
            if not hasattr(self, vname):
                if path is None:
                    path = self.h5fpath
                mynode, mypath  = self.find_variable_in_h5f(vname,path=path,return_path=True)
                if len(mynode)==0 and hasattr(self, 'create_'+vname) or kwargs.get('force',0):
                    if debug:
                        print(vname)
                    mynode = self.perform_create_and_save(vname, overwrite, path, **kwargs)
                elif len(mynode)>0:
                    if stage: # load data into memory
                        self.load(vname,path=mypath[0])
                        try:
                            mynode = getattr(self, vname)
                           # dir(self)
                        except: 
                            traceback.print_exc()
                else:
                    log.debug(errormsg+';'+vname+' not found in memory or in the hdf5 file')
                    mynode = None
            else:
                mynode = getattr(self, vname)
            mynodes.append(mynode)
        return mynodes[0] if len(mynodes)==1 else mynodes

    def perform_create_and_save(self,vname,overwrite=True,path='h5f.root',**kwargs):
        nodelist = getattr(self, 'create_'+vname)(**kwargs)
        mynode = getattr(self, vname)
        try:
            self.save(nodelist, overwrite ,path=path, **kwargs)
        except Exception as detail:
            print(detail)
            log.debug(detail)
        return mynode
        
    def managed_names(self):
        '''collects variable names from methods that start with 'create_'
        These data are managed by this class and transfer between hdf5 file and memory is done automatically'''
        import inspect
        return ['rawdata']+[method[7:] for method in self.__dict__ if inspect.ismethod(getattr(self, method)) and method.find('create_')>-1]
    
    def check_before_file_operation(self, names):
        '''various checks to be performed before saving/loading '''
        self.attrnames = list(set(self.attrnames)) # make sure entries are unique
        if names is not None and isinstance(names, str):
            names = [names]
        elif names is None:
            names = self.managed_names()+self.attrnames
        return names
        
    def save(self, names=None, overwrite=True, path=None, verify=False, filters = filters, **kwargs):
        ''' Saves data to the hdf5 file. If name is omitted then all data managed by this class will be saved.
           Numpy array, recarray,(list of) list of arrays,  and dict are supported.'''
        try:
            with self.write:
                if path is None: path = self.h5fpath
                names = self.check_before_file_operation(names)
                croot = introspect.traverse(self, path)
                if croot is None:
                    if kwargs.get('createparents', False) == True:
                        path1 = path.replace('.','/').replace('h5f/root','')
                        groupname = path1.split('/')[-1]
                        parent = path1.rstrip(groupname)
                        croot = self.h5f.create_group(parent, groupname, createparents = True)
                    else:
                        raise tables.exceptions.NoSuchNodeError('The specified path {} under which variables {} need to be saved does not exist. Specify the keyword argument "createparents" as True to automatically create the node specified in the path'.format((path), (names)))
                for vname in names:
                    hasit = self.find_variable_in_h5f(vname, path=path)
                    if hasit is None:
                        pass
                    if not hasattr(self, vname) and len(hasit)>0 and overwrite==False:
                        continue #item already in file but not in memory, nothing to do now
                    elif not hasattr(self, vname) and len(hasit)>0 and overwrite==True:
                        log.error('Tried to overwrite '+vname+' in hdf5 file but it is not in memory.')
                    if isinstance(getattr(self, vname), numbers.Number) or (isinstance(getattr(self, vname), str) and len(getattr(self,vname))<256):
                        #strings, int and float are saved as attribute
                        setattr(croot._v_attrs, vname, getattr(self,vname))
                        continue
                    if vname in hasit:
                        if overwrite:  # we cannot be sure that new content is the same size as old content
                            for vn1 in hasit:
                                # bug: hasit contains nodes different from root, but user is not obliged to provide the path for those nodes, thus removenode will not find the node in the root
                                try:
                                    self.h5f.remove_node(croot,vn1, recursive=True)
                                except Exception as e: #only if vn1 is stored as attribute?
                                    print(e)
                                    raise
                        else:
                            # if data exist in hdf5 file then return. If user explicitly told to overwrite it, then continue
                            continue
                    vp = getattr(self, vname)
                    hp = croot  #pointer to keep track where we are in the hdf hierarchy
                    if vp is None:
                        log.warning('Requested to save '+vname +' but it is not available in this object.')
                        continue
                    if isinstance(vp, list):
                        self.list2hdf(vp, vname,hp, filters, overwrite)
                    elif isinstance(vp, numpy.ndarray) or isinstance(vp, str) or isinstance(vp, bytes):
                        self.ndarray2hdf(vp, vname, hp, filters, overwrite, chunkshape=kwargs.get('chunkshape'))  # also handles e.g. string buffers created by cPickle.dumps
                    elif isinstance(vp, dict):
                        self.dict2hdf(vp, vname,hp, filters, overwrite)
                    else:
                        raise TypeError(vname+' cannot be saved, its type is unsupported')
                self.h5f.flush()
                if verify:
                    written = self.findvar(names)
                    self.close()
                    self.open_with_lock('a')
                    reread = self.findvar(names)
                    print('verified')
                    if written !=reread:
                        raise IOError('Reread data is not the same as written')
        except Exception as e:
            print(e)
            raise
        
    def save_vlarray(self, root, vn,  alist, filters=filters):
        myatom = tables.Atom.from_dtype(numpy.dtype(alist[0].dtype, (0, )*alist[0].ndim))
        vlarray = self.h5f.create_vlarray(root, vn, myatom,#(shape=()),
                                                 vn,
                                                 filters=filters)#VLarrays
        try:
            for item in alist:
                if item.ndim==0:
                    vlarray.append([item.tolist()])
                else:
                    vlarray.append(item)
        except:
            traceback.print_exc()
            pdb.set_trace()
    
    def list2hdf(self, vp, vn, hp, filters=filters, overwrite=False, chunkshape=None):
        if len(vp)==0: #empty list
            self.h5f.create_array(hp, vn, 'empty list', "empty list")
            return
        list_type = introspect.list_type(vp)
        if list_type == 'list_of_arrays':
            self.save_vlarray(hp, vn, vp, filters=filters)
        elif list_type == 'arrayized':
            vpa = numpy.array(vp)
            try:
                typepost = 'arrayized'
                isstr = is_decodable_to_str(vp[0])
                if isinstance(vp[0], str) or isstr:
                    typepost +='+string'
                self.saveCArray(vpa, vpa.shape, tables.Atom.from_dtype(vpa.dtype), hp, vn, overwrite, filters, typepost=typepost, chunkshape=chunkshape)
            except:
                raise
        elif 'recarrays' in list_type:# is 'list_of_recarrays': 
            # list of recarrays that have the same fields
            root = self.h5f.create_group(hp, vn, vn+'_'+list_type)
            for d in range(len(vp[0].dtype)):
                fname = vp[0].dtype.names[d]
                if 'uniform_shaped' in list_type:
                    numpyarray=numpy.squeeze([v[fname] for v in vp])
                    self.saveCArray(numpyarray, numpyarray.shape, tables.Atom.from_dtype(vp[0][fname].dtype), root, fname, overwrite, filters, typepost='_'+list_type, chunkshape=chunkshape)
                else: # each recarray in the list has different numbe of elements
                    self.save_vlarray(root, fname, [v[fname] for v in vp],filters=filters )
        elif list_type == 'list_of_lists':
            if len(vp)>10**Hdf5io.maxelem:
                raise NotImplementedError('Saving list of lists is supported till '+str(10**Hdf5io.maxelem)+' elements, increase x in the expression {0:0x} below and make sure old files will be read')
            root = self.h5f.create_group(hp, vn, vn+'_list_of_lists')
            for i in range(len(vp)):
                self.list2hdf(vp[i],Hdf5io.elemformatstr.format(i),root,filters,overwrite)
            if 0: # list type already detects if list of lsit is arrayizable so this below is deprected
               list_type = introspect.list_type(vp[0])
               if list_type == 'list_of_arrays': # list of lists of arrays
                    vlarray = self.h5f.create_vlarray(hp, vn,
                                                     tables.Atom.from_dtype(vp[0][0].dtype),#(shape=()),
                                                     vn,
                                                     filters=tables.Filters(1))#VLarrays
                    for item in vp:
                        shapes = [i.shape for i in item]
                        allequal = [s == shapes[0] for s in shapes]
                        if sum(allequal) == len(item): # all elements in the list have the same number of values
                            conc_item = numpy.r_[item]
                        else:
                            raise NotImplementedError('Saving list of lists of arrays when arrays have different shape is not implemented')
                        vlarray.append(conc_item)
                    setattr(self._v_attrs,vn+'_listlength',len(vp[0]))
        elif list_type == 'list_of_dicts': # list of dicts, all elements of the list must be a dict
            if len(vp)>10**Hdf5io.maxelem:
                raise NotImplementedError('Saving list of dicts is supported till '+str(10**Hdf5io.maxelem)+' elements, increase x in the expression {0:0x} below and make sure old files will be read')
            root = self.h5f.create_group(hp, vn, vn+'_list_of_dicts')
            for i0 in range(len(vp)):
                self.dict2hdf(vp[i0], Hdf5io.elemformatstr.format(i0),root, filters,  overwrite)
        elif list_type == 'inhomogenous_list':  # TODO: could use bloscpack directly? Estimate pickle performance depdneing on datadepth and size
            datastream = numpy.array(pickle.dumps(vp))
            self.ndarray2hdf(datastream, vn+'inhomogenous_list', hp)
                
                
    def ndarray2hdf(self, vp, vn, hp, filters=filters, overwrite=False, typepost='', chunkshape=None):
        #vp contains the pointer to the actual numpy ndarray
        if isinstance(vp, str) or isinstance(vp, bytes):
            vp = numpy.array(vp)
            typepost = '__String' if isinstance(vp, str) else '__Bytes'
        vdtype = vp.dtype#eval('self.'+vn+'.dtype')
        if vdtype.names is None:
            self.saveCArray(vp, vp.shape, tables.Atom.from_dtype(vp.dtype), hp, vn, overwrite, filters, typepost=typepost, chunkshape=chunkshape)
        else: # save recarray
            gn = vn
            cdtype = vdtype.fields
            fnames = list(cdtype.keys()) #names in hdf5 file's subgroup, e.g. self.h5f.root.quantified.full
            vnames = [gn+"['"+c+"']" for c in list(cdtype.keys())] # names in the current object e.g. self.quantified['full']
            vdtypes = [cdtype[d][0] for d in fnames]
            root = self.h5f.create_group(hp, gn, gn+'_recarray')
            for vn, fn, vdtype in zip(vnames,fnames,vdtypes):
                try:
                    atom = tables.Atom.from_dtype(vdtype)
                except:
                    raise RuntimeError('Cannot determine pytables atom from numpy data dtype.')
                cvp = getattr(vp.view(numpy.recarray), fn)
                vs = cvp.shape
                self.saveCArray(cvp, vs, atom, root, fn, overwrite, filters)
    
    def dict2hdf(self, vp, vn, hp, filters=filters, overwrite=False):
        '''Saves a python dict into the hdf5 file. There are restrictions on what elements a dict can have but
        this method can be updated to meet new needs.
        This methods tries to determine if an item in the dict is anything other than (array or list) of numeric values. 
        Array (or list) of numeric values is simply saved as CArray (or attribute), other types are saved calling
        the appropriate xxx2hdf method recursively.
        '''
        gn = vn
        fnames = list(vp.keys()) #names in hdf5 file's subgroup, e.g. self.h5f.root.quantified.full
        #vnames = [gn+"['"+str(c)+"']" for c in fnames] # names in the current object e.g. self.quantified['full']
        if overwrite and hasattr(hp,gn):
            self.h5f.remove_node(hp, gn, recursive=True)
        root = self.h5f.create_group(hp, gn, str(gn)+'_dict')
        for fn in fnames:
            if vp[fn] is None:
                vp[fn] = []
            if hasattr(vp[fn], 'keys'): 
                self.dict2hdf(vp[fn], fn,root, filters,  overwrite)
                continue
            if not isinstance(fn, str):
                from numbers import Number
                if isinstance(fn, numpy.ndarray) and sum(fn.shape)==0 or isinstance(fn, Number):
                    typepost = '__Number'
                else:
                    raise TypeError(f'Dict key {fn} with this type cannot be saved into hdf5 hierarchy')
            else:
                typepost=''
            if introspect.list_type(vp[fn]) is not None:
                self.list2hdf(vp[fn], fn ,root, filters,  overwrite)  # save as list of lists
                continue
            else:
                if not hasattr(vp[fn],'shape'):
                    vp[fn] = numpy.array(vp[fn])
            #atom = tables.Atom.from_dtype(vp[fn].dtype)
            vs = vp[fn].shape
            if len(vs)==0:
                setattr(root._v_attrs,str(fn)+typepost,vp[fn])
            else:
                self.ndarray2hdf(vp[fn], str(fn),root,  filters, overwrite, typepost=typepost)
#                self.saveCArray(vp[fn], vs, atom, root, fn,overwrite,filters)
                
    def saveCArray(self, vp, vs, atom, root, fn, overwrite, filters, typepost='', chunkshape=None):
        '''Saves a numpy array as CArray in the opened pytables file.
       vp is the reference to the actual data, vs is the shape of the array, root is the node under which the array has to be created
        fn is the name of the new node in the pytables file.'''
        if sum(vs)==0 or vp.size == 0: #0 dim array
            vla = self.h5f.create_vlarray(root, fn, atom, fn+typepost, filters=filters)
            if not hasattr(vp,'data') or vp.size==0: #empty array, create placeholder in h5f file
                return
            else:
                vla.append([vp])
                return
        if not overwrite and hasattr(root, fn):#vn in self.h5f.root.__members__:
            # if data exist in hdf5 file then return. If user explicitly told to overwrite it, then continue
            return
        if overwrite and hasattr(root, fn):#vn in self.h5f.root.__members__: # we cannot be sure that new content is the same size as old content
            self.h5f.remove_node(root,fn)
            self.h5f.flush()
        log.debug("writing " + fn)
        dimstr = ':,' * len(vs)
        if hasattr(vp, 'mask') and vp.mask is not numpy.ma.nomask:
            mask = vp.mask
            fnm = fn+'_mask'
            typepost += '_masked'
            if hasattr(root, fnm):
                self.h5f.remove_node(root, fnm)
            self.h5f.create_carray(root, fnm, tables.atom.BoolAtom(), vs, filters=filters, title=typepost)
            pcmd = "getattr(root,fnm)" + "[" + dimstr[:-1] + "] =mask[" + dimstr[:-1] + "]"
            exec((pcmd), locals())
            manode = self.h5f.get_node(root, fnm)
            setattr(manode._v_attrs, 'fill_value', vp.fill_value)
        self.h5f.create_carray(root, fn, atom, vs, filters=filters, title=typepost, chunkshape=chunkshape)
        pcmd = "getattr(root,fn)"+"["+dimstr[:-1]+"] =vp["+dimstr[:-1]+"]"
        try:
            exec((pcmd), locals())
        except:
            print((self.filename))

    def load(self, names=None, path=None):
        names = self.check_before_file_operation(names)
        if path is None:
            path = self.h5fpath
        successfully_loaded=[]
        with self.read:
            for vname in names:
                if hasattr(self.h5f.get_node(self.dot2slash(path))._v_attrs,vname): #data stored as attribute?
                    setattr(self,vname,getattr(self.h5f.get_node(self.dot2slash(path))._v_attrs,vname))
                    vp = getattr(self,vname)
                    setattr(self, vname, coerce_if_string23(vp))
                    successfully_loaded.append(vname)
                    continue
                if len(self.find_variable_in_h5f(vname,path=path))>0  and not self._want_abort: # data is in the file but not available in the class instance, load
                    log.debug("loading "+vname)
                    if not self._want_abort: 
                        self.load_variable(vname, path=path)
                        successfully_loaded.append(vname)
        return successfully_loaded

    def find_variable_in_h5f(self, vn, path = None, return_path=False, regexp=False):
        '''Tries to give back any leaf name in self.h5f that contain vn. Group nodes '''
        with self.read:
            if path is None:
                path = self.h5fpath
            if path.find('.')>-1:
                path = self.dot2slash(path)
            try:
                myroot = self.h5f.get_node(path)
            except:
                log.warning(path+' not found in '+self.filename)
                return [], [] if return_path else []
            #else: # maybe we look for a variable that is split up into multiple leafs?
            if hasattr(myroot._v_attrs, vn): 
                if return_path:
                    hpath = [self.slash2dot(path)]
                hasit = [vn] # variable is found as an attribute
            else:
                nodelist = self.h5f.list_nodes(path)
                if regexp:
                    ree = re.compile(vn)
                    if return_path:
                        hpath = [self.slash2dot(n._v_parent._v_pathname) for n in nodelist if len(re.findall(ree,n._v_name))>0]
                    hasit = [n._v_name for n in nodelist if len(re.findall(ree,n._v_name))>0]
                else:
                    if return_path:
                        hpath = [self.slash2dot(n._v_parent._v_pathname) for n in nodelist if n._v_name==vn]
                    hasit = [n._v_name for n in nodelist if n._v_name==vn]
            if return_path:
                return hasit,hpath
            else:
                return hasit
            
    def slash2dot(self, slashedpath):
        if slashedpath =='/': return 'h5f.root'
        else:
            path= slashedpath.replace('/','.')
            return ('h5f.root'+path)
        
    def dot2slash(self,dottedpath):
        '''converts a hdf5 path string from the format:
        'h5f.root' to '/'
        '''
        startnode = dottedpath.replace('.','/')
        startnode = startnode[startnode.index('root')+4:] or '/'
        return startnode
    
    def load_variable(self, vn, path=None):
        '''
        Loads the variable "vname" from the hdf5 file.
        '''
        try:
            if path is None:
                path = self.h5fpath
            croot = introspect.traverse(self, path)
            hasit = self.find_variable_in_h5f(vn,path=path)
            if len(hasit)>0:
                for vname in hasit:
                    if self.h5f.get_node(croot, vname)._v_title == 'empty list':
                        setattr(self, vname, [])
                        return
                    log.debug("reading "+vn+" in 1 chunk")
                    if isinstance(self.h5f.get_node(croot, vname),tables.Group):
                        # read a (eventually) nested data structure
                        group = self.h5f.get_node(croot, vname)
                        myvar = self.step_into_hdfgroup(group)
                        setattr(self,vname,myvar)
                    else:
                        if self.h5f.get_node(croot, vname).shape[0]==0: #saved empty list or empty array
                            setattr(self, vname, numpy.empty((0, ))) # simple read would return a list which is not the same as the empty numpy array saved in the file
                            return
                        myarray = self.read_array(croot, vname)
                        setattr(self, vname, myarray) # load array  from hdf5 file
                        title = self.h5f.get_node(croot, vname)._v_title
                        if '__String' in title or '__Bytes' in title:  # probably a long cPickle serialized string was stored in a 1-length VLArray, remove placeholder dimension
                            setattr(self, vname, getattr(self, vname)[0].tostring())
                        if 'inhomogenous' in title:
                            setattr(self, vname, pickle.loads(getattr(self, vname)))
                        if 'arrayized' in title:
                            if '+string' in title:  # py3: pytables stores strings as bytes, need to convert to string
                                setattr(self, vname, [item1.decode() for item1 in getattr(self, vname)])
                            else:
                                setattr(self, vname, [item1 for item1 in getattr(self, vname)])

                        # list of equal sized non-recarrays:
                        if isinstance(self.h5f.get_node(croot, vname),tables.VLArray) and hasattr(croot._v_attrs,vname+'_listlength'):
                            a = [numpy.split(item,getattr(croot._v_attrs,vname+'_listlength')) for item in getattr(self,vname)]
                            setattr(self,vname,a)
            else:
                log.debug("Variable "+vn+" not found in the cache file!")
                pass
        except Exception:
            import traceback
            print((traceback.format_exc()))
            log.debug("error reading "+vn+" from "+self.filename)
   
    def read_array(self, croot, vname):
        datanode = self.h5f.get_node(croot, vname)
        vname = datanode._v_name
        title = vname+'_'+datanode._v_title
        loadedarray = datanode.read()
        if 'masked' in title:
            mask = self.h5f.get_node(croot, vname + '_mask').read()
            vp1 = numpy.ma.array(loadedarray)
            vp1.mask = mask
            numpy.ma.set_fill_value(vp1, self.h5f.get_node(croot, vname + '_mask')._v_attrs.fill_value)
            loadedarray = vp1
        return loadedarray

    def hdf2dict(self, group):
        myvar = {}
        for n in self.h5f.iter_nodes(group):  # read data stored in arrays, walk trough names in hdf5 file's subgroup, e.g. self.h5f.root.quantified.full
            if n._v_name.replace('_mask','') in myvar:  # masked array's main data holder?
                continue  # mask has already been loaded via read_array
            if isinstance(n,tables.Group):
                myvar[n._v_name] = self.step_into_hdfgroup(n)
            else:
                try:
                    myvar[n._v_name] = self.read_array(group, n._v_name)  #n.read()
                except Exception as e:
                    print(e)
                    raise
                if 'arrayized' in n._v_title:
                    try:
                        if hasattr(myvar[n._v_name],'tolist'):
                            myvar[n._v_name] = myvar[n._v_name].tolist()
                        if '+string' in n._v_title:
                            myvar[n._v_name] = [item.decode('Latin1') for item in myvar[n._v_name]]
                    except:
                        pdb.set_trace()
                        raise
                try:
                    if isinstance(myvar[n._v_name], str) and myvar[n._v_name] =='empty list':
                        myvar[n._v_name]=[]
                except:
                    raise RuntimeError('')
        for n in group._v_attrs._f_list(): # read values stored as attribute
            typepos=  n.find('__')
            if typepos>0:
                mytype = n[typepos+2:]
                if mytype =='Number':
                    if '.' in n[:typepos]:
                        vname = float(n[:typepos]) # use float for ints too
                    else:
                        vname = int(n[:typepos])
                else:
                    raise TypeError('Type '+mytype+' not implemented to be stored in hdf5')
            else: vname=n
            myvar[vname] = getattr(group._v_attrs, n)
            myvar[vname] = coerce_if_string23(myvar[vname])
            if str(myvar[vname]) =='empty list': myvar[vname]=[]
        return myvar
                        
    def hdf2ndarray(self, group):
        mydtype = [n._v_name for n in self.h5f.iter_nodes(group)]
        try:
            myshape = getattr(group,mydtype[0]).shape
        except Exception as e:
            print(e)
        value_type = getattr(group,mydtype[0]).atom.dtype
        if 'recarrays' in group._v_title:
            listoflists= list(zip(*[getattr(group, d).read() for d in mydtype]))
            myvar = [numpy.rec.fromrecords(list(zip(*i)), names=mydtype) for i in listoflists] # na itt akad
        else: # pack out into one numpy ndarray
            myvar = numpy.empty(myshape, dtype=list(zip(mydtype,[value_type]*len(mydtype))))
            for nodename in mydtype:
                myvar[nodename] = self.h5f.get_node(group,nodename).read()

        return myvar
      
    def hdf2recarray(self,group):
        '''reads a record array from hdf5 file'''
        hasit = self.h5f.list_nodes(group)
        names = [n._v_name for n in hasit]
        firstnode = hasit[0]
        vdtype = numpy.dtype({'names':names, 'formats':[firstnode.atom.dtype.type]*len(hasit)})
        var0 = [] # empty list that will contain n lists that each contain numpy arrays of different length
        for v_i in range(len(hasit)):
            var0.append( hasit[v_i].read()) # load from hdf5 file
        #now we have all our fields, let's merge them
        var3 = list(zip(*var0)) # now we can slice as : var3[0][0] contains e.g. soma_roi_cols and var3[0][1] contains rows
        try:
            if 'uniform_shaped' in group._v_title:
                var2 = [numpy.array(item, dtype=vdtype) for item in var3]
            else:
                raise NotImplementedError('Cannot write this data structure into hdf5')
        except:
            var2 = [numpy.array(list(zip(*item[:])), dtype=vdtype) for item in var3]
        return var2   
        
    def hdf2list(self,group):
        listlength = len([n._v_name for n in self.h5f.iter_nodes(group) if n._v_name.isdigit()])
        mylist = [[]]*listlength
        for n in self.h5f.iter_nodes(group):
            if isinstance(n,tables.Group):
                mylist[int(n._v_name)] = self.step_into_hdfgroup(n)
            else:
                try:
                    mylist[int(n._v_name)] = n.read()
                except:
                    raise
                if 'inhomogenous' in n._v_title:
                    mylist[int(n._v_name)] = pickle.loads(mylist[int(n._v_name)])
                if 'arrayized' in n._v_title:
                    if '+string' in n._v_title:
                        mylist[int(n._v_name)] = numpy.char.decode(mylist[int(n._v_name)]).tolist()
                    else:
                        mylist[int(n._v_name)] = mylist[int(n._v_name)].tolist()
        return mylist
                            
    def step_into_hdfgroup(self, group):
        '''reads complex datatypes (dict, list, numpy recarray) from the hdf hierarchy
        '''
        gtitle = str(group._v_title)#.tostring()
        if gtitle.find('_dict')==len(group._v_title)-5:
            myvar = self.hdf2dict(group)
        elif 'recarrays' in group._v_title:
            myvar = self.hdf2recarray(group)
        elif gtitle.find('recarray')==len(group._v_title)-8:
            myvar=self.hdf2ndarray(group)
        elif 'list_of_' in gtitle:
            myvar = self.hdf2list(group)
        else: raise NotImplementedError(group._v_name+ ' has unknown data type in hdf5 file.')
        return myvar
        
    def cleanup(self):
        if hasattr(self, 'h5f') and self.h5f is not None and self.h5f.isopen:
            log.debug("object closing closes h5f")
            with lockman.lockpool[self.filelockkey].acquire(False):
                self.h5f.flush()
                self.h5f.close()
        log.debug("aborted, file closed "+self.filename)


    def attrsize(self, name):
        '''Gives back the size of the attribute "name" in bytes.'''
        n = getattr(self, name+'source')
        obj = introspect.traverse(self, n)
        if 'h5f' in n: #we check in the hdf5 file
            return numpy.cumprod(obj.shape)[-1] * obj.atom.dtype.itemsize
        else:
            return obj.size*obj.dtype.itemsize

def read_item(file1,  attrname, lockfile_path=None, file_mode = 'r'):
    '''Opens the hdf5file, reads the attribute and closes the file'''
    if hasattr(file1,'findvar'):
        h5f = file1
        value = h5f.findvar(attrname)
    else:
        if not os.path.exists(file1):
            raise OSError('Hdf5file '+file1+' not found')
        with closing(Hdf5io(file1, lockfile_path=lockfile_path, file_mode=file_mode)) as h5f:
            value = h5f.findvar(attrname)
    return value
    
def save_item(filename,  varname,  var, lockfile_path=None, overwrite = False):
    try:
        h = Hdf5io(filename,lockfile_path=lockfile_path)
        setattr(h,  varname,  var)
        h.save(varname,  overwrite = overwrite)
    except:
        raise OSError('Problem with hdf5file')
    finally:
        h.close()

