import importlib
import pkgutil
import sys
import inspect
'''
@author : Waleed Mohamed Ali
@Date   : 13/Jan/2018
'''
#"""
#reference :
# https://michaelheap.com/python-dynamically-load-all-modules-in-a-folder/
# https://stackoverflow.com/questions/3365740/how-to-import-all-submodules
#"""

def import_submodules(package, recursive=True):
	""" Import all submodules of a module, recursively, including subpackages

	:param package: package (name or actual module)
	:type package: str | module
	:rtype: dict[str, types.ModuleType]
	"""
	global __all__
	if isinstance(package, str):
		package = importlib.import_module(package)
	results = {}
	for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
		full_name = package.__name__ + '.' + name
		print('loading',name)
		results[full_name] = importlib.import_module(full_name)
		if recursive and is_pkg:
			results.update(import_submodules(full_name))
	return results
import_submodules(__name__)
