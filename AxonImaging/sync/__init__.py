try:
    from .sync import Sync
except (ImportError, IOError) as e:
    print("Import warning: %s." % e)
from .dataset import Dataset

__version__ = 1.3

#if i venture in the slipstream
#between the viaducts of your dreams
#where immobile steel rods crack
#and the ditch in the back roads stop
#could you find me
