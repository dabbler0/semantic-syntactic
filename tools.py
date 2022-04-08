import torch
import os

# Simple caching infrastructure
root_dir = '/raid/lingo/abau/slim-cache'
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

def get_filename(name, prop, version):
    if prop is None:
        return os.path.join(root_dir, name)
    else:
        return os.path.join(root_dir, '%s::%s::%d' % (name, prop, version))

class FlexiblePropElement:
    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index = index

    def __getitem__(self, prop):
        return self.dataset[prop][self.index]

class Dataset:
    def __init__(self, name):
        self.name = name
        self.loaded_properties = {}

    def get_property(self, prop):
        # TODO dependency versioning resolution

        # If already loaded in ram, return
        prop_string = '%s::%d' % (prop.name, prop.version)
        if prop_string not in self.loaded_properties:

            # If cached version in files, load
            fname = get_filename(self.name, prop.name, prop.version)
            if os.path.exists(fname):
                print(fname)
                self.loaded_properties[prop_string] = torch.load(fname)

            else:
                self.loaded_properties[prop_string] = prop.fn(self)
                torch.save(self.loaded_properties[prop_string], fname)

        return self.loaded_properties[prop_string]

    def __getitem__(self, index):
        if isinstance(index, Property):
            return self.get_property(index)
        elif isinstance(index, int):
            return FlexiblePropElement(self, index)

class Property:
    def __init__(self, dependencies, name, fn, version = 0):
        self.name = name
        self.dependencies = dependencies
        self.fn = fn
        self.version = version

class InjectedProperty(Property):
    def __init__(self, name, version = 0):
        super(InjectedProperty, self).__init__([], name, None, version)

def create_dataset(name, injected_props):
    # Invert the arrays of the various initial item
    for propname in injected_props:
        fname = get_filename(name, propname, 0)
        torch.save(injected_props[propname], fname)

    return Dataset(name)
