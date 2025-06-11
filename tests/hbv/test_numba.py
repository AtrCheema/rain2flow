
import os
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
site.addsitedir(package_path)

import unittest