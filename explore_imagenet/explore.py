# %%
from exin.exin import find_class
from pprint import pprint
import io

prefix = "n"
# %%
pprint(find_class(prefix + '00007846'), stream=io.open("cat_person.txt", "w"))
# %%
pprint(find_class(prefix + '14564367'), stream=io.open("cat_gun.txt", "w"))
# %%
