"""Generate relocatable documentation and samples when building a wheel."""
from pathlib import Path
import importlib.util

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithResources(build_py):
    def run(self):
        super().run()
        if self.editable_mode:
            return
        root = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location('_knotstudio_build_docs', root/'tools/build_docs.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = Path(self.build_lib)/'recognizer'/'_resources'
        module.build_bundle(root, target)

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        target = Path(self.build_lib)/'recognizer'/'_resources'
        outputs.extend(str(path) for path in target.rglob('*') if path.is_file())
        return outputs


setup(cmdclass={'build_py': BuildWithResources})
