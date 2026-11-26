"""Block model management (no Qt): the protected binary block model file (.gvbm) and block models made from a
solid or imported from a Surpac ASCII export. (Attribute editing and grade estimation are GeoViewer's.)

Modules:

* ``store``      - the encrypted, compressed .gvbm file format
* ``build``      - new block model inside a solid, Surpac ASCII import
* ``attributes`` - attribute descriptions and statistics (and the helpers used when building)
* ``parallel``   - long tasks in worker processes (blocks inside a solid)
"""

BLOCK_MODEL_EXT = ".gvbm"
